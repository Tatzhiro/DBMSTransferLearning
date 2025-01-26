import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import shutil
from IPython import embed
from regression import group_features
from copy import deepcopy
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import warnings
from sklearn.exceptions import ConvergenceWarning


@hydra.main(version_base=None, config_path="conf/transfer_learning")
def main(cfg: DictConfig) -> None:
    sys_argv = [x for x in sys.argv if "+" not in x]    
    args_dict = dict([sys.argv[(n*2)+1:(n*2)+3] for n in range(int(len(sys_argv) / 2))])
    assert "--config-name" in args_dict
    config_name = os.path.basename(args_dict["--config-name"]).split(".")[0]
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)

    models = cfg.models
    sizes = cfg.sizes
    num_trial = cfg.num_trial

    df_mean = pd.DataFrame(index=sizes)
    df_std = pd.DataFrame(index=sizes)
    for key in models:
        print(f"validation start: {models[key].name}")
        pipeline = instantiate(cfg.models[key].pipeline)
        trials = {size: [] for size in sizes}
        for i in range(num_trial):
            print(f"\ttrial = {i}")
            pipeline.seed = i
            pipeline.initialize()
            for size in sizes:
                print(f"\ttrain_size = {size}")
                pipeline.simulate(size)
                target_df = pipeline.target_df
                prediction = pipeline.predict(target_df)
                try:
                    mape = mean_absolute_percentage_error(target_df[pipeline.system.get_perf_metric()], prediction)
                except:
                    mape = np.nan
                print(f"\t\tmape = {mape}")
                trials[size].append(mape)
        mean_scores = [np.nanmean(trials[size]) for size in sizes]
        mean_stds = [np.std(trials[size]) for size in sizes]
        df_mean[f"{models[key].name}"] = mean_scores
        df_std[f"{models[key].name}"] = mean_stds
        print(f"validation done: {models[key].name}\n")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_runtime_output_dir = hydra_cfg["runtime"]["output_dir"]
    output_name = os.path.join(hydra_runtime_output_dir, config_name)

    df_mean.to_csv(f"{output_name}.csv")
    df_std.to_csv(f"{output_name}_std.csv")

    plot_design = instantiate(cfg.plot_design)
    instantiate(cfg.plot_function, df_std, plot_design, f"{output_name}_std.pdf")

    config_output_dir = f"{hydra_runtime_output_dir}/../../transfer_learning/{config_name}"
    shutil.copytree(hydra_runtime_output_dir, config_output_dir, dirs_exist_ok=True)

def custom_sort_key(index_str):
    # Split the index string into parts
    parts = index_str.split('_')
    # Convert parts to floats for numerical comparison
    floats = [float(part) for part in parts]
    return floats
    
if __name__ == "__main__":
    main()