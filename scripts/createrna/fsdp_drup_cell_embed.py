#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSDP 离线推理（3B）：支持可选化学编码，提取细胞嵌入。
示例：
  torchrun --standalone --nproc_per_node=8 scripts/inference/fsdp_predict_3b_cell_emb_chem.py \
    --adata /data/input.h5ad \
    --gene-id-key ensembl_id \
    --model-dir /models/tx-3b-v2 \
    --batch-size 8 \
    --seq-len 1024 \
    --precision bf16 \
    --use-chem-inf \
    --drug-key drug \
    --save /data/output_tx3b_chem.h5ad
"""

import os
import argparse
import logging
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
)

from tahoe_x1.model import TXModel
from tahoe_x1.model.blocks import TXBlock
from tahoe_x1.tokenizer import GeneVocab
from tahoe_x1.utils.util import loader_from_adata
from tahoe_x1.data.dataloader import CountDataset
from tahoe_x1.data.collator import DataCollator

log = logging.getLogger("fsdp_infer_chem")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)


class CountDatasetWithDrug(CountDataset):
    def __init__(self, count_matrix, gene_ids, drugs, cls_token_id=None, pad_value=None):
        super().__init__(count_matrix=count_matrix, gene_ids=gene_ids, cls_token_id=cls_token_id, pad_value=pad_value)
        self.drugs = drugs

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["drug"] = self.drugs[idx]
        return item


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata", type=str, required=True)
    p.add_argument("--gene-id-key", type=str, default="ensembl_id")
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--use-chem-inf", action="store_true")
    p.add_argument("--drug-key", type=str, default="drug")
    p.add_argument("--drug-to-id-path", type=str, default=None)
    return p.parse_args()


def init_distributed():
    backend = "nccl" if (torch.cuda.is_available() and os.name != "nt") else "gloo"
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


def build_fsdp_model(model_cfg, collator_cfg, precision: str) -> TXModel:
    attn_impl = model_cfg.get("attn_config", {}).get("attn_impl", "flash")
    if attn_impl == "triton":
        model_cfg["attn_config"]["attn_impl"] = "flash"
        model_cfg["attn_config"]["use_attn_mask"] = False

    want_flash = model_cfg.get("attn_config", {}).get("attn_impl", "flash") == "flash"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        if want_flash:
            log.warning("flash-attn 未安装，回退到 torch 注意力实现。")
            model_cfg["attn_config"]["attn_impl"] = "torch"
            model_cfg["attn_config"]["use_attn_mask"] = False

    model_cfg["return_gene_embeddings"] = True
    model_cfg["do_mlm"] = False
    model_cfg["init_device"] = "meta"

    model = TXModel(model_config=model_cfg, collator_config=collator_cfg, device=None)

    if precision == "bf16":
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
        amp_dtype = torch.float16
    else:
        mp = None
        amp_dtype = torch.float32

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=lambda m, *_: isinstance(m, TXBlock),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params=True,
        sync_module_states=True,
    )
    setattr(fsdp_model, "_amp_dtype", amp_dtype)
    return fsdp_model


def load_weights_all_ranks(fsdp_model: TXModel, weights_path: str, strict: bool = True):
    sd = load_file(weights_path)
    # 清理前缀
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
    fsdp_model.load_state_dict(cleaned, strict=strict)


def build_loader(
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
    use_chem: bool,
    drug_key: str,
):
    adata = sc.read_h5ad(adata_path)

    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var[gene_id_key]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    genes = adata.var[gene_id_key].tolist()
    gene_ids = np.array([vocab[g] for g in genes], dtype=int)

    adata_rank = adata[rank::world_size, :]

    cfg = om.merge(collator_cfg, om.create({"do_mlm": False}))

    if use_chem:
        # 保证前两位为 <cls> 和 <drug>
        cfg["keep_first_n_tokens"] = max(cfg.get("keep_first_n_tokens", 1), 2)
        cfg["use_chem_token"] = True
        # 构造药物列表
        if drug_key in adata_rank.obs:
            drugs = (
                adata_rank.obs[drug_key].astype(str).replace({"nan": "<pad>"}).fillna("<pad>").tolist()
            )
        else:
            drugs = ["<pad>" for _ in range(adata_rank.n_obs)]

        count_matrix = adata_rank.X
        if isinstance(count_matrix, np.ndarray):
            pass
        else:
            try:
                count_matrix = count_matrix.toarray()
            except Exception:
                from scipy.sparse import csr_matrix
                count_matrix = csr_matrix(count_matrix).toarray()

        dataset = CountDatasetWithDrug(
            count_matrix=count_matrix,
            gene_ids=gene_ids,
            drugs=drugs,
            cls_token_id=vocab["<cls>"],
            pad_value=cfg["pad_value"],
        )
        collate_fn = DataCollator(
            vocab=vocab,
            drug_to_id_path=cfg.get("drug_to_id_path", None),
            do_padding=cfg.get("do_padding", True),
            unexp_padding=False,
            pad_token_id=cfg.pad_token_id,
            pad_value=cfg.pad_value,
            do_mlm=False,
            do_binning=cfg.get("do_binning", True),
            log_transform=cfg.get("log_transform", False),
            target_sum=cfg.get("target_sum"),
            mlm_probability=cfg.mlm_probability,
            mask_value=cfg.mask_value,
            max_length=seq_len or len(gene_ids),
            sampling=cfg.sampling,
            num_bins=cfg.get("num_bins", 51),
            right_binning=cfg.get("right_binning", False),
            keep_first_n_tokens=cfg.get("keep_first_n_tokens", 2),
            use_chem_token=True,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
    else:
        adata_rank = adata[rank::world_size, :]
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
    return adata, adata_rank, loader, gene_ids, cfg


def infer_cell_embs(fsdp_model: TXModel, loader, collator_cfg) -> np.ndarray:
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    fsdp_model.eval()

    amp_dtype = getattr(fsdp_model, "_amp_dtype", torch.float32)
    outs: List[torch.Tensor] = []
    with torch.no_grad(), torch.amp.autocast(enabled=(amp_dtype != torch.float32), dtype=amp_dtype, device_type=device.type):
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            out = fsdp_model(
                genes=batch["gene"],
                values=batch["expr"],
                gen_masks=batch["gen_mask"],
                key_padding_mask=~batch["gene"].eq(collator_cfg.pad_token_id),
                drug_ids=(batch["drug_ids"] if "drug_ids" in batch else None),
                skip_decoders=True,
            )
            outs.append(out["cell_emb"].to("cpu").to(dtype=torch.float32))
    arr = torch.cat(outs, dim=0).numpy()
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr


def main():
    args = parse_args()
    rank, world_size = init_distributed()
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Rank {rank}/{world_size} on {device} initialized.")

    paths = resolve_model_paths(args.model_dir)
    vocab = GeneVocab.from_file(paths["vocab"])
    model_cfg = om.load(paths["model_cfg"])  # DictConfig
    collator_cfg = om.load(paths["collator_cfg"])  # DictConfig

    # 配置化学编码
    use_chem = bool(args.use_chem_inf)
    if use_chem:
        if args.drug_to_id_path:
            collator_cfg["drug_to_id_path"] = args.drug_to_id_path
        elif collator_cfg.get("drug_to_id_path") is None:
            collator_cfg["drug_to_id_path"] = {
                "remote": "s3://tahoe-hackathon-data/MFM/drug_to_id_pad.json",
                "local": "drug_to_id_pad.json",
            }
        collator_cfg["use_chem_token"] = True
        collator_cfg["keep_first_n_tokens"] = max(collator_cfg.get("keep_first_n_tokens", 1), 2)
        if "chemical_encoder" not in model_cfg:
            log.warning("模型配置缺少 chemical_encoder, 无法启用化学编码, 改为 use_chem_inf=False")
            use_chem = False

    fsdp_model = build_fsdp_model(model_cfg, collator_cfg, args.precision)
    load_weights_all_ranks(fsdp_model, paths["weights"], strict=True)
    log.info("FSDP model weights loaded.")

    adata_full, adata_rank, loader, gene_ids, coll_cfg = build_loader(
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
        use_chem=use_chem,
        drug_key=args.drug_key,
    )

    log.info(f"Rank{rank}: 开始推理 …")
    cell_arr_local = infer_cell_embs(fsdp_model, loader, coll_cfg)

    gathered: List[Optional[np.ndarray]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, cell_arr_local)

    if rank == 0:
        cell_arr = np.concatenate(gathered, axis=0)
        log.info(f"Collected cell embeddings shape: {cell_arr.shape}")
        if args.save:
            adata_full = sc.read_h5ad(args.adata)
            model_name = os.path.basename(os.path.abspath(args.model_dir))
            adata_full.obsm[model_name] = cell_arr
            adata_full.write_h5ad(args.save)
            log.info(f"Saved embeddings to {args.save}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
