import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import shutil
from IPython import embed


@hydra.main(version_base=None, config_path="conf", config_name="plot.yaml")
def main(cfg: DictConfig) -> None:
    df = pd.read_csv(cfg.csv_path, index_col=0)
    plot_type = cfg.plot_type
    evaluation = cfg.evaluations[plot_type]
    for plot in evaluation.keys():
        plot_function = evaluation[plot].function
        design = instantiate(evaluation[plot].design)
        instantiate(plot_function, df, design, f"{cfg.output_path}")

if __name__ == "__main__":
    main()