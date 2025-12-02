import os,sys
sys.path.insert(0, os.path.abspath('..'))
from scripts.createrna.drug_cell_embed import predict_embeddings_single_stream
from omegaconf import OmegaConf as om

cfg = {
  "model_name": "Tx1-3b",
  "paths": {
    "adata_input": "stromal.h5ad",
    "model_dir": "model/3b-model",
    "adata_output": "./pred_stromal.h5ad",
    #"drug_to_id_path": "drug_to_id_pad.json" #不涉及扰物扰动，注释掉
  },
  "data": {
    "cell_type_key": "celltype_define",
    "gene_id_key": "ensembl_id",
    #"drug_key": "drug", #不涉及扰物扰动，一般直接注释掉即可
  },
  "predict": {
    "batch_size": 32,
    "seq_len_dataset": 1024, #基因数
    "num_workers": 8,
    "prefetch_factor": 8,
    "return_gene_embeddings": False,
    "use_chem_inf": False, #关闭化学编码设为False
    "skip_decoders": False,
  },
}
cfg = om.create(cfg)
adata = predict_embeddings_single_stream(cfg)

