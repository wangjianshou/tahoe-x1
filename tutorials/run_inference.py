import os
import sys
import argparse
from omegaconf import OmegaConf as om
from tahoe_x1.createrna.drug_cell_embed import predict_embeddings_single_stream

def main():
    parser = argparse.ArgumentParser(description="Run cell embedding prediction with external YAML config")
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    cfg = om.load(args.config)
    adata = predict_embeddings_single_stream(cfg)

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath('..'))
    main()

