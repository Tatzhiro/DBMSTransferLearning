from regression import ParameterImportanceRetrieval, MySQLConfiguration
import wandb
import os

data_dir = "dataset/transfer_learning/mysql/chimera_tech"
target_csv = os.path.join(data_dir, "88c190g-result.csv")
workloads = [
    "64-1000000-4-oltp_read_only-0.2",
    "64-1000000-4-oltp_read_only-0.6",
    "64-1000000-4-oltp_write_only-0.2",
    "64-1000000-4-oltp_write_only-0.6",
    "64-1000000-4-oltp_read_write_50-0.2",
    "64-1000000-4-oltp_read_write_50-0.6",
    "10-4-4-tpcc-nan",
    "100-4-4-tpcc-nan",
]

def study(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        similarity = ParameterImportanceRetrieval(
            system=MySQLConfiguration(),
            data_dir=data_dir,
            train_dir="dataset/metric_learning"
        )
        similarity.model = None
        similarity.set_hyper_parameters(config=cfg)
        error, result = similarity.loocv(target_csv, workloads)
        wandb.log({"error": error})
        data = [[k, v] for k, v in result.items()]
        table = wandb.Table(data=data, columns=["workload_label", "error"])
        wandb.log({"error_chart": table})
        

def main():
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "error",
            "goal": "minimize"
        },
        "parameters": {
            "num_hidden_layer": {
                "values": [10, 20, 30, 50]
            },
            "hidden_size": {
                "values": [256, 128, 64]
            },
            "batch_size": {
                "values": [256, 128, 64, 32]
            },
            "learning_rate": {
                "values": [0.0001, 0.0005, 0.001]
            },
            "num_epochs": {
                "values": [30, 50, 100]
            },
            "weight_decay": {
                "values": [1e-5, 1e-4, 1e-3]
            }
        }
    }
    
    project = "ParameterImportanceSimilarity"
    sweep_id = wandb.sweep(sweep_config, project=project)

    wandb.agent(sweep_id, study)
    wandb.finish()
    
if __name__ == "__main__":
    main()