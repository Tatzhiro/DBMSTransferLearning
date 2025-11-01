import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from regression import Pipeline, group_features
from sklearn.metrics import mean_absolute_percentage_error
import shutil
from IPython import embed
from copy import deepcopy


@hydra.main(version_base=None, config_path="conf/parameter_space")
def main(cfg: DictConfig) -> None:
    sys_argv = [x for x in sys.argv if "+" not in x]    
    args_dict = dict([sys.argv[(n*2)+1:(n*2)+3] for n in range(int(len(sys_argv) / 2))])
    assert "--config-name" in args_dict
    config_name = os.path.basename(args_dict["--config-name"]).split(".")[0]

    models = cfg.models
    parameter_space = list(cfg.parameters)
    dfs = {}
    for key in models:
        pipeline: Pipeline = instantiate(cfg.models[key].pipeline)
        model_name = cfg.models[key].name

        system = pipeline.system
        pipeline.initialize()
        pipeline.simulate(cfg.iterations)
        target_df = pipeline.target_df
        prediction = pipeline.predict()
        mape = mean_absolute_percentage_error(target_df[pipeline.system.get_perf_metric()], prediction)
        print(f"mape = {mape}")

        predict_df = deepcopy(target_df)
        prediction = pipeline.predict()
        predict_df["tps"] = prediction

        target_df = group_features(target_df, parameter_space, system)
        predict_df = group_features(predict_df, parameter_space, system)
        target_df = target_df.set_index("config")
        predict_df = predict_df.set_index("config")

        df = pd.merge(target_df, predict_df, on="config")
        df = df.rename(columns={f"{system.get_perf_metric()}_x": "Actual Value", f"{system.get_perf_metric()}_y": f"{model_name} Prediction"})
        sorted_index = sorted(df.index, key=custom_sort_key)
        df = df.reindex(sorted_index)
        dfs[model_name] = df
    
    # each df in dfs is a DataFrame with the actual and predicted values
    # create a combined DataFrame with the actual and predicted values for each model
    combined_df = pd.DataFrame()
    for model_name, df in dfs.items():
        if combined_df.empty:
            combined_df = df[["Actual Value", f"{model_name} Prediction"]]
        else:
            combined_df = pd.merge(combined_df, df[f"{model_name} Prediction"], left_index=True, right_index=True)
    # sort based on Actual Value
    # combined_df = combined_df.sort_values(by="Actual Value")
    

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_runtime_output_dir = hydra_cfg["runtime"]["output_dir"]
    output_name = os.path.join(hydra_runtime_output_dir, config_name)

    combined_df.to_csv(f"{output_name}.csv")

    config_output_dir = f"{hydra_runtime_output_dir}/../../parameter_space/{config_name}"
    shutil.copytree(hydra_runtime_output_dir, config_output_dir, dirs_exist_ok=True)

def custom_sort_key(index_str):
    # Split the index string into parts
    parts = index_str.split('_')
    # Convert parts to floats for numerical comparison
    floats = [float(part) for part in parts]
    return floats
    
if __name__ == "__main__":
    main()