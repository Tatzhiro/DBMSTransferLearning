import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import shutil


@hydra.main(version_base=None, config_path="conf/cv")
def main(cfg: DictConfig) -> None:
    sys_argv = [x for x in sys.argv if "+" not in x]    
    args_dict = dict([sys.argv[(n*2)+1:(n*2)+3] for n in range(int(len(sys_argv) / 2))])
    assert "--config-name" in args_dict
    config_name = os.path.basename(args_dict["--config-name"]).split(".")[0]

    models = cfg.models
    sizes = cfg.sizes
    if "data" in cfg.keys():
        domain_data = cfg.data["train"]
        target_data = cfg.data["target"]

    df = pd.DataFrame(index=sizes)
    for key in models:
        print(f"cross validation start: {models[key].name}")
        pipeline = instantiate(cfg.models[key].pipeline)
        scores = []
        if "data" in cfg.models[key].keys():
            domain_data = cfg.models[key].data["train"]
            target_data = cfg.models[key].data["target"]
        for size in sizes:
            print(f"\ttrain_size = {size}")
            scores.append(pipeline.cross_validation(domain_data, target_data, train_size=size))
        df[f"{models[key].name}"] = scores
        print(f"cross validation done: {models[key].name}\n")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_runtime_output_dir = hydra_cfg["runtime"]["output_dir"]
    output_name = os.path.join(hydra_runtime_output_dir, config_name)

    df.to_csv(f"{output_name}.csv")

    plot_design = instantiate(cfg.plot_design)
    instantiate(cfg.plot_function, df, plot_design, f"{output_name}.pdf")

    config_output_dir = f"{hydra_runtime_output_dir}/../../cv/{config_name}"
    shutil.copytree(hydra_runtime_output_dir, config_output_dir, dirs_exist_ok=True)
    
if __name__ == "__main__":
    main()