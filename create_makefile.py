import os

def make_rules(directory, plot_type, script):
    for filename in os.listdir(directory):
        if ".yaml" not in filename:
            continue
        conf_name = filename.removesuffix(".yaml")
        # * make df
        target = os.path.join("outputs", plot_type, conf_name, f"{conf_name}.csv")
        sources=" ".join([os.path.join(directory, conf_name + ".yaml"), f"scripts/{script}"])
        rule = f"{target}: {sources}\n\tpython scripts/{script} --config-name {conf_name}.yaml"
        return target, rule, conf_name

# Step 1: Load conf file
all_targets = []
rules = []
scripts = {"cv": "scripts/cross_validation.py", 
           "parameter_space": "scripts/parameter_space.py", 
           "correlation": "scripts/correlation.py",
           "transfer_learning": "scripts/transfer_learning.py",
           "important_parameter": "scripts/important_parameter.py",
           "real_data": "scripts/real_data.py"}
targets = {"cv": [], "cv_plot": [],
           "parameter_space": [], "parameter_space_plot": [],
           "transfer_learning": [], "transfer_learning_plot": [],
           "real_data": [],
           "correlation": [],
           "important_parameter": []}
plot_script = "scripts/plot.py"

for dir in os.listdir("scripts/conf"):
    if ".yaml" in dir: continue
    directory = f"scripts/conf/{dir}"
    csv_script = scripts[dir]
    for filename in os.listdir(directory):
        if ".yaml" not in filename: continue
        conf_name = filename.removesuffix(".yaml")
        # * make df
        csv_target = os.path.join("outputs", dir, conf_name, f"{conf_name}.csv")
        csv_sources=" ".join([os.path.join(directory, conf_name + ".yaml"), csv_script])
        csv_rule = f"{csv_target}: {csv_sources}\n\tpython -u {csv_script} --config-name {conf_name}.yaml"
        targets[dir].append(csv_target)
        all_targets.append(csv_target)
        rules.append(csv_rule)

        if targets.get(f"{dir}_plot") != None:
            plot_target = os.path.join("outputs", dir, conf_name, f"{conf_name}.pdf")
            plot_sources=" ".join([csv_target, plot_script, csv_script, "scripts/conf/plot.yaml"])
            plot_rule = f"{plot_target}: {plot_sources}\n\tpython {plot_script} ++csv_path={csv_target} ++plot_type={dir} ++output_path={plot_target}"
            all_targets.append(plot_target)
            rules.append(plot_rule)
            targets[f"{dir}_plot"].append(plot_target)


# Step 3: Write Makefile
with open("./makefile", "w") as f:
    f.write("export PYTHONPATH=$PYTHONPATH:.\n")
    f.write("export HYDRA_FULL_ERROR=1\n\n")
    f.write("all: {}\n\n".format(" ".join(all_targets)))
    for tgt in targets:
        f.write(f"all_{tgt}: {' '.join(targets[tgt])}\n\n")
    f.write("\n\n".join(rules))
    f.write("\n\nclean:\n\trm -rf outputs/*")