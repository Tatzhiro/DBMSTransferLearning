import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import shutil
from IPython import embed
from regression import Pipeline
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

import sys


def _safe_std(values):
    arr = np.asarray(values, dtype=float)
    n = np.count_nonzero(~np.isnan(arr))
    if n <= 1:
        return 0.0  # no dispersion with <=1 valid value
    return float(np.nanstd(arr, ddof=1))


@hydra.main(version_base=None, config_path="conf/optimum")
def main(cfg: DictConfig) -> None:
    sys_argv = [x for x in sys.argv if "+" not in x]
    args_dict = dict([sys.argv[(n*2)+1:(n*2)+3] for n in range(int(len(sys_argv) / 2))])
    assert "--config-name" in args_dict
    config_name = os.path.basename(args_dict["--config-name"]).removesuffix(".yaml")
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_runtime_output_dir = hydra_cfg["runtime"]["output_dir"]
    output_name = os.path.join(hydra_runtime_output_dir, config_name)

    models = cfg.models
    sample_sizes = cfg.sizes
    num_trial = cfg.num_trial

    # DataFrame to hold mean & std columns for each model
    df_stats = pd.DataFrame(index=sample_sizes)

    for key in models:
        model_name = f"{models[key].name}"
        print(f"validation start: {model_name}")
        pipeline: Pipeline = instantiate(cfg.models[key].pipeline)
        trials = simulate(pipeline, sample_sizes, num_trial)

        # aggregate per size across trials
        mean_scores = [float(np.nanmean(trials[size])) for size in sample_sizes]
        std_scores  = [_safe_std(trials[size]) for size in sample_sizes]

        df_stats[f"{model_name}_mean"] = mean_scores
        df_stats[f"{model_name}_std"]  = std_scores

        print(f"validation done: {model_name}\n")

    df_stats.index.name = "train_size"
    df_stats.to_csv(f"{output_name}.csv")

    config_output_dir = f"{hydra_runtime_output_dir}/../../optimum/{config_name}"
    shutil.copytree(hydra_runtime_output_dir, config_output_dir, dirs_exist_ok=True)


def simulate(pipeline, sample_sizes, num_trial=5):
    trials = {size: [] for size in sample_sizes}
    for i in range(num_trial):
        print(f"\ttrial = {i}")
        pipeline.seed = i
        pipeline.initialize()
        last_result = 0
        for size in sample_sizes:
            print(f"\ttrain_size = {size}")
            if last_result == 1.0:
                print("\t\tSkip remaining sizes since the optimum is already found.")
                trials[size].append(last_result)
                continue
            pipeline.simulate(size)
            target_df = pipeline.target_df
            prediction = pipeline.predict()
            pred_argmax = np.argmax(prediction)
            rec_config = target_df.iloc[pred_argmax]
            actl_perf = target_df.iloc[pred_argmax][pipeline.system.get_perf_metric()]
            actl_max = target_df[pipeline.system.get_perf_metric()].max()
            ratio = actl_perf / actl_max if actl_max != 0 else np.nan
            print(f"\t\trecommended config: {rec_config.to_dict()}, predicted performance: {prediction[pred_argmax]}")
            print(f"\t\tactual performance: {actl_perf}, actual max performance: {actl_max}, ratio: {ratio}")
            last_result = ratio
            trials[size].append(last_result)
    return trials


def custom_sort_key(index_str):
    parts = index_str.split('_')
    floats = [float(part) for part in parts]
    return floats


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)


if __name__ == "__main__":
    main()
