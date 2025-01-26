import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from regression import FeatureSelector, read_data_csv, group_features
from sklearn.linear_model import LinearRegression
import shutil
from IPython import embed
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


@hydra.main(version_base=None, config_path="conf/correlation")
def main(cfg: DictConfig) -> None:
    sys_argv = [x for x in sys.argv if "+" not in x]    
    args_dict = dict([sys.argv[(n*2)+1:(n*2)+3] for n in range(int(len(sys_argv) / 2))])
    assert "--config-name" in args_dict
    config_name = os.path.basename(args_dict["--config-name"]).split(".")[0]

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_runtime_output_dir = hydra_cfg["runtime"]["output_dir"]
    
    feature_selector: FeatureSelector = instantiate(cfg.feature_selector)
    for i in range(len(cfg.datadir)):
      for j in range(i+1, len(cfg.datadir)):
        system_x = instantiate(cfg.systems[i])
        system_y = instantiate(cfg.systems[j])
        for filename_x in os.listdir(cfg.datadir[i]):
          for filename_y in os.listdir(cfg.datadir[j]):
            domain_path = f"{cfg.datadir[i]}/{filename_x}"
            target_path = f"{cfg.datadir[j]}/{filename_y}"
            domain_name = f"{cfg.datadir[i].split('/')[-1]}_{filename_x.split('.')[0]}"
            target_name = f"{cfg.datadir[j].split('/')[-1]}_{filename_y.split('.')[0]}"

            df_x = read_data_csv(domain_path, system_x, cfg.workload)
            df_y = read_data_csv(target_path, system_y, cfg.workload)

            important_parameters = feature_selector.select_important_features(df_x, system_x)

            assert(system_x.get_perf_metric() == system_y.get_perf_metric())
            x = group_features(df_x, important_parameters, system_x)
            y = group_features(df_y, important_parameters, system_y)

            real_df = pd.merge(x, y, on="config")
            real_df["features"] = [important_parameters for _ in range(len(real_df))]
            x_label = instantiate(cfg.axis_label, domain_name)
            y_label = instantiate(cfg.axis_label, target_name)
            output_name = os.path.join(hydra_runtime_output_dir, f"{x_label} vs {y_label}")

            plot_design = instantiate(cfg.plot_design, x_label, y_label)
            column_name = system_x.get_perf_metric()
            instantiate(cfg.plot_function, real_df, column_name, plot_design, f"{output_name}")

    config_output_dir = f"{hydra_runtime_output_dir}/../../correlation/{config_name}"
    shutil.copytree(hydra_runtime_output_dir, config_output_dir, dirs_exist_ok=True)
    
if __name__ == "__main__":
    main()