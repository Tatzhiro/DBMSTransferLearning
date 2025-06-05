import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from regression import read_data_csv, group_features, drop_unimportant_parameters
import shutil
from IPython import embed
from copy import deepcopy
import warnings
import json


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
    workloads = cfg.workloads
    manual_default_param_values = cfg.manual_default_param_values
    # get a set of values that are different from the original default values
    changed_param_values = {}
    for key in manual_default_param_values.keys():
        if system.default_param_values[key] != manual_default_param_values[key]:
            changed_param_values[key] = manual_default_param_values[key]
    # set the new default values
    system.default_param_values = manual_default_param_values
    
    for workload in workloads:
        for filename in os.listdir(cfg.data_path):
            if filename.endswith(".csv"):
                output_name = os.path.join(hydra_runtime_output_dir, f"{filename.split('.')[0]}")
                os.makedirs(output_name, exist_ok=True)
                data_df = read_data_csv(os.path.join(cfg.data_path, filename), system, workload)
                parameter_space = list(cfg.parameters)
                if cfg.all_parameters:
                    df = group_features(data_df, parameter_space, system)
                    df = df.set_index("config")

                    sorted_index = sorted(df.index, key=custom_sort_key)
                    df = df.reindex(sorted_index)
                    df.to_csv(f"{output_name}.csv")
                    instantiate(cfg.plot_function, df, plot_design, f"{output_name}/all: {' '.join(parameter_space)}")
                for i, p in enumerate(parameter_space):
                    for j, q in enumerate(parameter_space):
                        if i > j:
                            continue
                        if i == j:
                            with warnings.catch_warnings(record=True) as w:
                                warnings.simplefilter("always")
                                df = group_features(data_df, [p], system)
                                if w:
                                    continue
                            df = df.set_index("config")
                            sorted_index = sorted(df.index, key=custom_sort_key)
                            df = df.reindex(sorted_index)
                            df.to_csv(f"{output_name}/{p}.csv")
                            # df = df.reset_index()
                            instantiate(cfg.plot_function, df, plot_design, f"{output_name}/{p}")
                        else:
                            perf = system.get_perf_metric()
                            df = drop_unimportant_parameters(data_df, [p, q], system)
                            df = df.groupby([p, q])[perf].mean().reset_index()
                            df.to_csv(f"{output_name}/{p}_{q}.csv", index=False)
                            x = df[p].values
                            y = df[q].values
                            
                            z = df[perf].values

                            # Create grid for interpolation
                            pivot = df.pivot(index=p, columns=q, values=perf)

                            # Create meshgrid from the actual values used
                            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
                            Z = pivot.values

                            # Plot
                            fig = plt.figure(figsize=(8, 6))
                            ax = fig.add_subplot(111, projection='3d')

                            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.5)

                            ax.set_xlabel(f"P{i}", labelpad=10)
                            ax.set_ylabel(f"P{j}", labelpad=10)
                            ax.set_zlabel("Throughput", labelpad=10)

                            ax.view_init(elev=30, azim=-135)

                            plt.tight_layout()
                            plt.savefig(f"{output_name}/{p}_{q}.pdf", bbox_inches='tight')
                            plt.close()
                    
                

        config_output_dir = f"{hydra_runtime_output_dir}/../../real_data/{workload}/{json.dumps(changed_param_values).replace(' ', '')}"
        shutil.copytree(hydra_runtime_output_dir, config_output_dir, dirs_exist_ok=True)

def custom_sort_key(index_str):
    # Split the index string into parts
    parts = index_str.split('_')
    # Convert parts to floats for numerical comparison
    floats = [float(part) for part in parts]
    return floats
    
if __name__ == "__main__":
    main()