import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from regression import read_data_csv
from sklearn.linear_model import LinearRegression
import shutil
from IPython import embed
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


@hydra.main(version_base=None, config_path="conf/important_parameter")
def main(cfg: DictConfig) -> None:
    sys_argv = [x for x in sys.argv if "+" not in x]    
    args_dict = dict([sys.argv[(n*2)+1:(n*2)+3] for n in range(int(len(sys_argv) / 2))])
    assert "--config-name" in args_dict
    config_name = os.path.basename(args_dict["--config-name"]).split(".")[0]
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_runtime_output_dir = hydra_cfg["runtime"]["output_dir"]
    data_dict = {}
    assert(len(cfg.datadir) == len(cfg.systems))
    for i in range(len(cfg.datadir)):
      dirname = cfg.datadir[i].split("/")[-1]
      system = instantiate(cfg.systems[i])
      for filename in os.listdir(cfg.datadir[i]):
        if filename.endswith(".csv"):
          df = read_data_csv(os.path.join(cfg.datadir[i], filename), system, cfg.workload)
          feature_selector = instantiate(cfg.feature_selector)
          feature_selector.select_important_features(df, system)
          important_parameters = feature_selector.get_parameter_importance()
          data_dict[f"{dirname}_{filename.split('.')[0]}"] = important_parameters
          
    df = pd.DataFrame(list(data_dict.items()), columns=["data", "important_parameters"])
    df.to_csv(f"{hydra_runtime_output_dir}/{config_name}.csv", index=False)

    config_output_dir = f"{hydra_runtime_output_dir}/../../important_parameter/{config_name}"
    shutil.copytree(hydra_runtime_output_dir, config_output_dir, dirs_exist_ok=True)
    
if __name__ == "__main__":
    main()