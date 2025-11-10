import numpy as np
import torch
import scanpy as sc
from omegaconf import DictConfig
from tahoe_x1.utils.util import load_model, loader_from_adata

def predict_embeddings_single_stream(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = device.type == "cuda"
    use_bf16 = use_autocast and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # config inputs
    model_name = cfg.get("model_name", "Tahoe-x1")
    cell_type_key = cfg.data.cell_type_key
    gene_id_key = cfg.data.gene_id_key
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
        # 将权重转换为 bf16/fp16 以减少显存（仅推理）
        model.model.to(dtype=autocast_dtype)
    model.eval()

    # load and align AnnData
    adata = sc.read_h5ad(cfg.paths.adata_input)
    adata = adata[~adata.obs[cell_type_key].isna(), :]
    adata.var["id_in_vocab"] = [
        vocab[g] if g in vocab else -1 for g in adata.var[gene_id_key]
    ]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    genes = adata.var[gene_id_key].tolist()
    gene_ids = np.array([vocab[g] for g in genes], dtype=int)

    # build loader
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
                    out = model.forward(batch, skip_decoders=cfg.predict.get('skip_decoders', False))
            else:
                out = model.forward(batch, skip_decoders=cfg.predict.get('skip_decoders', False))

            # stream cell embeddings to CPU
            cell_embs.append(out["cell_emb"].detach().cpu().numpy())

            # optional: stream gene embeddings and aggregate
            if return_gene_embeddings:
                gene_ids_b = out["gene_ids"].detach().cpu()
                gene_emb_b = out["gene_emb"].detach().cpu()
                b, s, d = gene_emb_b.shape
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
