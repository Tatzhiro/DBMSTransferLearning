import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from regression import read_data_csv
from regression import group_features
import shutil
from IPython import embed
from copy import deepcopy


@hydra.main(version_base=None, config_path="conf/real_data")
def main(cfg: DictConfig) -> None:
    sys_argv = [x for x in sys.argv if "+" not in x]    
    args_dict = dict([sys.argv[(n*2)+1:(n*2)+3] for n in range(int(len(sys_argv) / 2))])
    assert "--config-name" in args_dict
    config_name = os.path.basename(args_dict["--config-name"]).split(".")[0]
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_runtime_output_dir = hydra_cfg["runtime"]["output_dir"]

    system = instantiate(cfg.system)
    plot_design = instantiate(cfg.plot_design)
    for filename in os.listdir(cfg.data_path):
        if filename.endswith(".csv"):
            output_name = os.path.join(hydra_runtime_output_dir, f"{filename.split('.')[0]}")
            os.makedirs(output_name, exist_ok=True)
            data_df = read_data_csv(os.path.join(cfg.data_path, filename), system, cfg.workload)
            parameter_space = list(cfg.parameters)
            if cfg.all_parameters:
                df = group_features(data_df, parameter_space, system)
                df = df.set_index("config")

                sorted_index = sorted(df.index, key=custom_sort_key)
                df = df.reindex(sorted_index)


                df.to_csv(f"{output_name}.csv")
                instantiate(cfg.plot_function, df, plot_design, f"{output_name}/all: {' '.join(parameter_space)}")
            for parameter in parameter_space:
                df = group_features(data_df, [parameter], system)
                df = df.set_index("config")
                sorted_index = sorted(df.index, key=custom_sort_key)
                df = df.reindex(sorted_index)
                df.to_csv(f"{output_name}/{parameter}.csv")
                instantiate(cfg.plot_function, df, plot_design, f"{output_name}/{parameter}")
                
            

    config_output_dir = f"{hydra_runtime_output_dir}/../../real_data/{cfg.workload}"
    shutil.copytree(hydra_runtime_output_dir, config_output_dir, dirs_exist_ok=True)

def custom_sort_key(index_str):
    # Split the index string into parts
    parts = index_str.split('_')
    # Convert parts to floats for numerical comparison
    floats = [float(part) for part in parts]
    return floats
    
if __name__ == "__main__":
    main()