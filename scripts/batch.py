import subprocess
import yaml
import shutil
import os
from time import time
import argparse


def process_workload(yaml_name):
    yaml_file = os.path.basename(yaml_name).removesuffix(".yaml")
    os.makedirs(f"outputs/transfer_learning/{name}/logs", exist_ok=True)
    command = f"make outputs/transfer_learning/{yaml_file}/{yaml_file}.pdf"
    command += f" 2>&1 | tee outputs/transfer_learning/{name}/logs/{yaml_file}.log"
    subprocess.run(command, check=True, shell=True)
    
    # Rename the output file for this workload.
    subprocess.run(f"cp outputs/transfer_learning/{yaml_file}/{yaml_file}.pdf outputs/transfer_learning/{name}/{yaml_file}.pdf", check=True, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Select the transfer mode to run."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["context_retrieval", "full_transfer", "data_transfer"],
        default="full_transfer",
        help="Operation mode (default: %(default)s). Choose from %(choices)s."
    )
    args = parser.parse_args()

    # You can now branch on args.mode:
    if args.mode == "context_retrieval":
        print("Running context retrieval…")
    elif args.mode == "full_transfer":
        print("Running full transfer…")
    elif args.mode == "data_transfer":
        print("Running data transfer…")
        
    name = args.mode
    
    workloads = [
        "10-4-4-tpcc-nan",
        "100-4-4-tpcc-nan",
        "64-1000000-4-oltp_write_only-0.2",
        "64-1000000-4-oltp_write_only-0.6",
        "64-1000000-4-oltp_read_only-0.2",
        "64-1000000-4-oltp_read_only-0.6",
        "64-1000000-4-oltp_read_write_50-0.2",
        "64-1000000-4-oltp_read_write_50-0.6",
    ]
    
    yamls = []
    
    for wl in workloads:
        original_yaml_path = f"scripts/conf/transfer_learning/{name}.yaml"
        # Create a temporary YAML file specific for this workload
        temp_yaml_path = f"scripts/conf/transfer_learning/{name}_{wl}.yaml"
        shutil.copyfile(original_yaml_path, temp_yaml_path)

        # Load and update the temporary YAML configuration.
        with open(temp_yaml_path, "r") as file:
            config = yaml.safe_load(file)
        config["workload"] = wl
        if "oltp" in wl:    
            config["plot_design"]["x_label"] = "Sample Size\n(1 sample = 120 sec)"
        else:
            config["plot_design"]["x_label"] = "Sample Size\n(1 sample = 330 sec)"
        with open(temp_yaml_path, "w") as file:
            yaml.dump(config, file)

        yamls.append(temp_yaml_path)
        
    subprocess.run("python create_makefile.py", shell=True, check=True)
    
    # Run each workload in parallel.
    time_table = {}
    for i, yaml_name in enumerate(yamls):
        print(f"Processing workload: {yaml_name}")
        start_time = time()
        process_workload(yaml_name)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {yaml_name}: {elapsed_time:.2f} seconds")
        time_table[yaml_name] = elapsed_time
        print(f"Done with {yaml_name} ({i+1}/{len(yamls)})")
    
    print("Time table for each workload:")
    for yaml_name, elapsed_time in time_table.items():
        print(f"{yaml_name}: {elapsed_time:.2f} seconds")
    print(f"Sum of elapsed times: {sum(time_table.values()):.2f} seconds")
        
