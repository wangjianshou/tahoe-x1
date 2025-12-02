import numpy as np
import torch
import scanpy as sc
from omegaconf import DictConfig
from scipy.sparse import csr_matrix, csc_matrix

from tahoe_x1.utils.util import load_model, loader_from_adata
from tahoe_x1.data.dataloader import CountDataset
from tahoe_x1.data.collator import DataCollator


class CountDatasetWithDrug(CountDataset):
    """扩展 CountDataset，使每个样本携带药物名以便化学编码。"""
    def __init__(self, count_matrix, gene_ids, drugs, add_cls_token=True, cls_token_id=None, pad_value=None):
        super().__init__(count_matrix=count_matrix, gene_ids=gene_ids,
                         add_cls_token=add_cls_token, cls_token_id=cls_token_id, pad_value=pad_value)
        self.drugs = drugs

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["drug"] = self.drugs[idx]  # collator中将用drug_to_id映射
        return item


def predict_embeddings_single_stream(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = device.type == "cuda"
    use_bf16 = use_autocast and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # config inputs
    model_name = cfg.get("model_name", "Tahoe-x1")
    cell_type_key = cfg.data.get('cell_type_key', None)
    gene_id_key = cfg.data.gene_id_key
    drug_key = cfg.data.get("drug_key", "drug")  # AnnData.obs中的药物列名
    return_gene_embeddings = cfg.predict.get("return_gene_embeddings", False)
    use_chem_inf = cfg.predict.get("use_chem_inf", None)

    batch_size = cfg.predict.get("batch_size", 16)
    max_length = cfg.predict.get("seq_len_dataset", 1024)
    num_workers = cfg.predict.get("num_workers", 8)
    prefetch_factor = cfg.predict.get("prefetch_factor", 8)
    adata_output_path = cfg.paths.get("adata_output", None)

    # load model and vocab
    model_dir = cfg.paths.get("model_dir")
    model, vocab, _, coll_cfg = load_model(
        model_dir=model_dir,
        device=device,
        return_gene_embeddings=return_gene_embeddings,
        use_chem_inf=use_chem_inf,
    )
    if use_autocast:
        model.model.to(dtype=autocast_dtype)  # 推理态混合精度，降低显存
    model.eval()

    # load and align AnnData
    adata = sc.read_h5ad(cfg.paths.adata_input)
    if cell_type_key and cell_type_key in adata.obs.columns:
        adata = adata[~adata.obs[cell_type_key].isna(), :]
    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var[gene_id_key]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    genes = adata.var[gene_id_key].tolist()
    gene_ids = np.array([vocab[g] for g in genes], dtype=int)

    # 是否启用化学编码（需要每样本drug）
    use_chem = bool(use_chem_inf)
    if use_chem:
        # 保证前两位为<cls>和<drug>
        coll_cfg["keep_first_n_tokens"] = max(coll_cfg.get("keep_first_n_tokens", 1), 2)
        coll_cfg["use_chem_token"] = True
        # 确保有drug_to_id映射路径
        if coll_cfg.get("drug_to_id_path", None) is None:
            local_json = cfg.paths.get("drug_to_id_path", "drug_to_id_pad.json")
            coll_cfg["drug_to_id_path"] = {
                "remote": "s3://tahoe-hackathon-data/MFM/drug_to_id_pad.json",
                "local": local_json,
            }

    # build loader
    if use_chem:
        # 构造CSR矩阵与药物列表
        count_matrix = adata.X
        if isinstance(count_matrix, np.ndarray):
            count_matrix = csr_matrix(count_matrix)
        elif isinstance(count_matrix, csc_matrix):
            count_matrix = count_matrix.tocsr()
        elif hasattr(count_matrix, "to_memory"):
            count_matrix = count_matrix.to_memory().tocsr()

        if drug_key in adata.obs:
            drugs = (
                adata.obs[drug_key]
                .astype(str)
                .replace({"nan": "<pad>"})
                .fillna("<pad>")
                .tolist()
            )
        else:
            # 没有药物列则全部用<pad>（不使用化学信息）
            drugs = ["<pad>" for _ in range(adata.n_obs)]

        dataset = CountDatasetWithDrug(
            count_matrix=count_matrix,
            gene_ids=gene_ids,
            drugs=drugs,
            cls_token_id=vocab["<cls>"],
            pad_value=coll_cfg["pad_value"],
        )
        collate_fn = DataCollator(
            vocab=vocab,
            drug_to_id_path=coll_cfg.get("drug_to_id_path", None),
            do_padding=coll_cfg.get("do_padding", True),
            unexp_padding=False,
            pad_token_id=coll_cfg.pad_token_id,
            pad_value=coll_cfg.pad_value,
            do_mlm=False,
            do_binning=coll_cfg.get("do_binning", True),
            log_transform=coll_cfg.get("log_transform", False),
            target_sum=coll_cfg.get("target_sum"),
            mlm_probability=coll_cfg.mlm_probability,
            mask_value=coll_cfg.mask_value,
            max_length=max_length,
            sampling=coll_cfg.sampling,
            num_bins=coll_cfg.get("num_bins", 51),
            right_binning=coll_cfg.get("right_binning", False),
            keep_first_n_tokens=coll_cfg.get("keep_first_n_tokens", 2),
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
        loader = loader_from_adata(
            adata=adata,
            collator_cfg=coll_cfg,
            vocab=vocab,
            batch_size=batch_size,
            max_length=max_length,
            gene_ids=gene_ids,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    # streaming inference
    cell_embs = []
    sums = None
    counts = None
    pad_token_id = coll_cfg["pad_token_id"]

    with torch.no_grad():
        for batch in loader:
            # move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = model.forward(batch, skip_decoders=True)
            else:
                out = model.forward(batch, skip_decoders=True)

            # stream cell embeddings to CPU
            cell_embs.append(out["cell_emb"].detach().cpu().numpy())

            # optional: stream gene embeddings and aggregate
            if return_gene_embeddings:
                gene_ids_b = out["gene_ids"].detach().cpu()
                gene_emb_b = out["gene_emb"].detach().cpu()
                _, _, d = gene_emb_b.shape
                if sums is None:
                    sums = np.zeros((len(vocab), d), dtype=np.float32)
                    counts = np.zeros((len(vocab),), dtype=np.float32)
                flat_ids = gene_ids_b.reshape(-1).numpy()
                flat_embs = gene_emb_b.reshape(-1, d).numpy()
                valid = flat_ids != pad_token_id
                np.add.at(sums, flat_ids[valid], flat_embs[valid])
                np.add.at(counts, flat_ids[valid], 1.0)

    # finalize cell embeddings
    cell_array = np.concatenate(cell_embs, axis=0)
    cell_array = cell_array / (np.linalg.norm(cell_array, axis=1, keepdims=True) + 1e-12)
    adata.obsm[model_name] = cell_array

    # finalize gene embeddings (if requested)
    if return_gene_embeddings:
        means = np.divide(
            sums, counts[:, None],
            out=np.ones_like(sums) * np.nan,
            where=counts[:, None] != 0,
        )
        adata.varm[model_name] = means[gene_ids, :]

    if adata_output_path is not None:
        adata.write_h5ad(adata_output_path)

    return adata
