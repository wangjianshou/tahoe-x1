#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSDP 多卡离线推理：加载本地 3B 模型，提取 cell embeddings（CLS）。
用法示例：
  torchrun --standalone --nproc_per_node=8 scripts/inference/fsdp_predict_3b_cell_emb.py \
    --adata /data/input.h5ad \
    --gene-id-key ensembl_id \
    --model-dir /models/tx-3b-v2 \
    --batch-size 8 \
    --seq-len 1024 \
    --precision bf16 \
    --save /data/output_tx3b.h5ad

要求 model-dir 目录下至少包含以下文件：
  - vocab.json
  - model_config.yml
  - collator_config.yml
  - model.safetensors
"""

import os
import argparse
import logging
import time
from typing import Dict, Any, List, Optional

import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
from omegaconf import OmegaConf as om
from safetensors.torch import load_file
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    FullStateDictConfig,
)

from tahoe_x1.model import TXModel
from tahoe_x1.model.blocks import TXBlock
from tahoe_x1.tokenizer import GeneVocab
from tahoe_x1.utils.util import loader_from_adata

log = logging.getLogger("fsdp_infer_offline")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata", type=str, required=True, help="输入 AnnData .h5ad 文件路径")
    p.add_argument("--gene-id-key", type=str, default="ensembl_id", help="adata.var 中基因 ID 列名")
    p.add_argument("--model-dir", type=str, required=True, help="本地模型目录（包含 vocab.json 等文件）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=1024, help="最大序列长度")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--save", type=str, default=None, help="保存到 .h5ad 路径（可选）")
    return p.parse_args()


def init_distributed():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def resolve_model_paths(model_dir: str) -> Dict[str, str]:
    files = {
        "vocab": os.path.join(model_dir, "vocab.json"),
        "model_cfg": os.path.join(model_dir, "model_config.yml"),
        "collator_cfg": os.path.join(model_dir, "collator_config.yml"),
        "weights": os.path.join(model_dir, "model.safetensors"),
    }
    for k, p in files.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少文件: {k} -> {p}")
    return files


def build_fsdp_model(model_cfg, collator_cfg, precision: str, strict_load_flag_holder: Dict[str, bool]) -> TXModel:
    # 推理时关闭 MLM；注意力实现兼容性修正
    attn_impl = model_cfg.get("attn_config", {}).get("attn_impl", "flash")
    if attn_impl == "triton":
        model_cfg["attn_config"]["attn_impl"] = "flash"
        model_cfg["attn_config"]["use_attn_mask"] = False

    # 可选：检查是否安装 flash-attn，否则回退
    want_flash = model_cfg.get("attn_config", {}).get("attn_impl", "flash") == "flash"
    has_flash = False
    try:
        import flash_attn  # noqa: F401
        has_flash = True
    except Exception:
        has_flash = False
    if want_flash and not has_flash:
        log.warning("flash-attn 未安装，回退到 torch 注意力实现。")
        model_cfg["attn_config"]["attn_impl"] = "torch"
        model_cfg["attn_config"]["use_attn_mask"] = False

    # 禁用化学编码（若数据中无 drug_ids）
    strict_load_flag_holder["strict"] = True
    if collator_cfg.get("use_chem_token", False):
        log.warning("推理中禁用化学编码(use_chem_token=False)。")
        collator_cfg["use_chem_token"] = False
        # 同步移除与化学编码相关的配置，满足 DataCollator 断言
        if "drug_to_id_path" in collator_cfg:
            del collator_cfg["drug_to_id_path"]
        if "chemical_encoder" in model_cfg:
            del model_cfg["chemical_encoder"]
        strict_load_flag_holder["strict"] = False

    # 返回基因嵌入以便后续扩展（当前仅提取 cell_emb）
    model_cfg["return_gene_embeddings"] = True
    model_cfg["do_mlm"] = False
    model_cfg["init_device"] = "meta"  # 关键：避免构造时实体化权重，利于分片加载

    # 构建模型（meta 设备）
    model = TXModel(model_config=model_cfg, collator_config=collator_cfg, device=None)

    # 精度设置
    if precision == "bf16":
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
        amp_dtype = torch.float16
    else:
        mp = None
        amp_dtype = torch.float32

    # FSDP 包裹：仅包裹 Transformer Block
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=lambda module, recurse, nonwrapped_numel: isinstance(module, TXBlock),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params=True,
        sync_module_states=True,
    )
    # 把 amp dtype 作为属性传递，供推理时使用
    setattr(fsdp_model, "_amp_dtype", amp_dtype)
    return fsdp_model


def fsdp_load_sharded_weights(fsdp_model: TXModel, weights_path: str, strict_load: bool):
    full_sd = None
    if dist.get_rank() == 0:
        log.info("Rank0: 正在从 safetensors 加载完整 state_dict …")
        t0 = time.time()
        full_sd = load_file(weights_path)
        t1 = time.time()
        try:
            total_bytes = sum(v.numel() * v.element_size() for v in full_sd.values())
            log.info("Rank0: 读盘完成，keys=%d, 大小≈%.2f GB, 用时=%.2fs", len(full_sd), total_bytes / (1024**3), t1 - t0)
        except Exception:
            log.info("Rank0: 读盘完成，keys=%d, 用时=%.2fs", len(full_sd), t1 - t0)

    def _clean_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prefixes = [
            "_fsdp_wrapped_module.",
            "module.",
            "model.",
            "net.",
            "tx_model.",
            "txmodel.",
        ]
        cleaned = {}
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            cleaned[nk] = v
        return cleaned

    # 统一采用非严格加载，避免 rank 之间因异常导致不同步
    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True),
    ):
        if dist.get_rank() == 0:
            cleaned_sd = _clean_keys(full_sd)
            t2 = time.time()
            fsdp_model.load_state_dict(cleaned_sd, strict=False)
            t3 = time.time()
            log.info("Rank0: state_dict 非严格加载+广播完成，用时=%.2fs", t3 - t2)
        else:
            log.info("Rank%d: 等待接收 Rank0 的广播 …", dist.get_rank())
            t2 = time.time()
            fsdp_model.load_state_dict({}, strict=False)
            t3 = time.time()
            log.info("Rank%d: 接收完成，用时=%.2fs", dist.get_rank(), t3 - t2)
    dist.barrier()
    log.info("所有 rank 已通过 barrier，同步完成。")


def build_loader_per_rank(
    adata_path: str,
    gene_id_key: str,
    vocab: GeneVocab,
    collator_cfg: Dict[str, Any],
    batch_size: int,
    seq_len: Optional[int],
    num_workers: int,
    prefetch_factor: int,
    world_size: int,
    rank: int,
):
    adata = sc.read_h5ad(adata_path)

    # 基因 ID → vocab 索引
    adata.var["id_in_vocab"] = [
        vocab[g] if g in vocab else -1 for g in adata.var[gene_id_key]
    ]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]  # 仅保留 vocab 中的基因
    genes = adata.var[gene_id_key].tolist()
    gene_ids = np.array([vocab[g] for g in genes], dtype=int)

    # 每个 rank 处理一部分细胞（简单按行步进切分）
    adata_rank = adata[rank::world_size, :]

    # collator 修正：推理关闭 MLM，保持 DictConfig 类型以支持属性访问
    cfg = om.merge(collator_cfg, om.create({"do_mlm": False}))

    loader = loader_from_adata(
        adata=adata_rank,
        collator_cfg=cfg,
        vocab=vocab,
        batch_size=batch_size,
        max_length=seq_len,
        gene_ids=gene_ids,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    log.info("Rank%d: 数据切片 cells=%d, genes=%d, batch_size=%d", rank, adata_rank.n_obs, adata_rank.n_vars, batch_size)
    return adata, adata_rank, loader, gene_ids, cfg


def infer_cell_embs(fsdp_model: TXModel, loader, collator_cfg, amp_dtype) -> np.ndarray:
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    fsdp_model.eval()

    cell_embs: List[torch.Tensor] = []
    first_batch = True
    with torch.no_grad(), torch.amp.autocast(enabled=(amp_dtype != torch.float32), dtype=amp_dtype, device_type=device.type):
        for data_dict in loader:
            ids = data_dict["gene"].to(device)
            expr = data_dict["expr"].to(device)
            gen_mask = data_dict["gen_mask"].to(device)
            src_key_padding_mask = ~ids.eq(collator_cfg.pad_token_id)

            out = fsdp_model(
                genes=ids,
                values=expr,
                gen_masks=gen_mask,
                key_padding_mask=src_key_padding_mask,
                drug_ids=(data_dict["drug_ids"].to(device) if "drug_ids" in data_dict else None),
                skip_decoders=True,
            )
            if first_batch:
                log.info("首个 batch 完成，cell_emb 形状=%s", tuple(out["cell_emb"].shape))
                first_batch = False
            cell_embs.append(out["cell_emb"].to("cpu").to(dtype=torch.float32))

    cell_array = torch.cat(cell_embs, dim=0).numpy()
    # 归一化（与现有脚本一致）
    cell_array = cell_array / np.linalg.norm(cell_array, axis=1, keepdims=True)
    return cell_array


def main():
    args = parse_args()
    rank, world_size = init_distributed()
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Rank {rank}/{world_size} on {device} initialized.")

    # 解析本地模型文件路径
    paths = resolve_model_paths(args.model_dir)

    # 加载 vocab 与配置
    vocab = GeneVocab.from_file(paths["vocab"])
    model_cfg = om.load(paths["model_cfg"])  # DictConfig
    collator_cfg = om.load(paths["collator_cfg"])  # DictConfig

    # 构建并分片加载模型
    strict_holder = {"strict": True}
    fsdp_model = build_fsdp_model(model_cfg, collator_cfg, args.precision, strict_holder)
    fsdp_load_sharded_weights(fsdp_model, paths["weights"], strict_holder["strict"])
    log.info("FSDP model weights loaded.")

    # 每个 rank 构建本地数据加载器
    adata_full, adata_rank, loader, gene_ids, coll_cfg = build_loader_per_rank(
        adata_path=args.adata,
        gene_id_key=args.gene_id_key,
        vocab=vocab,
        collator_cfg=collator_cfg,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        world_size=world_size,
        rank=rank,
    )

    # 推理提取 cell embeddings
    amp_dtype = getattr(fsdp_model, "_amp_dtype", torch.float32)
    log.info("Rank%d: 开始推理 …", rank)
    cell_arr_local = infer_cell_embs(fsdp_model, loader, coll_cfg, amp_dtype)

    # 汇总到 rank0
    gathered: List[Optional[np.ndarray]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, cell_arr_local)

    if rank == 0:
        cell_arr = np.concatenate(gathered, axis=0)
        log.info(f"Collected cell embeddings shape: {cell_arr.shape}")
        if args.save:
            adata_full = sc.read_h5ad(args.adata)
            # 用 model-dir 的最后一级目录名作为 obsm key
            model_name = os.path.basename(os.path.abspath(args.model_dir))
            adata_full.obsm[model_name] = cell_arr
            adata_full.write_h5ad(args.save)
            log.info(f"Saved embeddings to {args.save}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
