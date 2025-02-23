import subprocess
import yaml

with open('workload.yaml') as f:
    workload = yaml.load(f, Loader=yaml.FullLoader)
    