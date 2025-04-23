import subprocess
import yaml
import shutil
import os
from concurrent.futures import ProcessPoolExecutor

name = "chimera_tech"

def process_workload(yaml_name):
    yaml_file = os.path.basename(yaml_name).removesuffix(".yaml")
    os.makedirs(f"outputs/transfer_learning/{name}/logs", exist_ok=True)
    subprocess.run(f"make outputs/transfer_learning/{yaml_file}/{yaml_file}.pdf 2>&1 | tee outputs/transfer_learning/{name}/logs/{yaml_file}.log", check=True, shell=True)
    

    # Rename the output file for this workload.
    subprocess.run(f"cp outputs/transfer_learning/{yaml_file}/{yaml_file}_barplot.pdf outputs/transfer_learning/{name}/{yaml_file}.pdf", check=True, shell=True)


if __name__ == '__main__':
    workloads = [
        "64-1000000-4-oltp_read_only-0.2",
        "64-1000000-4-oltp_read_only-0.6",
        "64-1000000-4-oltp_read_write_50-0.2",
        "64-1000000-4-oltp_read_write_50-0.6",
        "64-1000000-4-oltp_write_only-0.2",
        "64-1000000-4-oltp_write_only-0.6",
        "10-4-4-tpcc-nan",
        "100-4-4-tpcc-nan",
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
        with open(temp_yaml_path, "w") as file:
            yaml.dump(config, file)

        yamls.append(temp_yaml_path)
        
    subprocess.run("python create_makefile.py", shell=True, check=True)
    
    # Run each workload in parallel.
    try:
        with ProcessPoolExecutor() as executor:
            executor.map(process_workload, yamls)
    except KeyboardInterrupt:
        print("Process interrupted.")
        os._exit(0)
        
