import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from regression import Pipeline, SystemConfiguration, MultivariateDataLoader, MultivariateRegressionPipeline, SimpleRegressionDataLoader, FeatureSelector, FeatureEngineer
from regression import group_features
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

    pipeline: Pipeline = instantiate(cfg.model.pipeline)
    model_name = cfg.model.name
    parameter_space = list(cfg.parameters)

    if type(pipeline) == MultivariateRegressionPipeline:
        data_loader = pipeline.data_loader
        system = data_loader.system
        train_data = cfg.data["train"]
        target_data = cfg.data["target"]
        if "few_shot" in cfg.keys():
            pipeline.fit_with_target(train_data, target_data, cfg.few_shot.sample_size)
        else:
            pipeline.fit(train_data)

        # To compare parameter space including training data
        # target_df = data_loader.load_training_df(train_data + [target_data])
        
        target_df = data_loader.make_data(target_data)
    else:
        system = pipeline.system
        pipeline.initialize()
        pipeline.simulate(cfg.iterations)
        target_df = pipeline.target_df
        prediction = pipeline.predict(target_df)
        mape = mean_absolute_percentage_error(target_df[pipeline.system.get_perf_metric()], prediction)
        print(f"mape = {mape}")

    predict_df = deepcopy(target_df)
    prediction = pipeline.predict(target_df)
    predict_df["tps"] = prediction

    target_df = group_features(target_df, parameter_space, system)
    predict_df = group_features(predict_df, parameter_space, system)
    target_df = target_df.set_index("config")
    predict_df = predict_df.set_index("config")

    df = pd.merge(target_df, predict_df, on="config")
    df = df.rename(columns={f"{system.get_perf_metric()}_x": "Actual Value", f"{system.get_perf_metric()}_y": f"{model_name} Prediction"})
    sorted_index = sorted(df.index, key=custom_sort_key)
    df = df.reindex(sorted_index)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_runtime_output_dir = hydra_cfg["runtime"]["output_dir"]
    output_name = os.path.join(hydra_runtime_output_dir, config_name)

    df.to_csv(f"{output_name}.csv")

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