export PYTHONPATH=$PYTHONPATH:.
export HYDRA_FULL_ERROR=1

all: outputs/important_parameter/mysql_rpl/mysql_rpl.csv outputs/real_data/mysql_rpl/mysql_rpl.csv outputs/real_data/mysql/mysql.csv outputs/correlation/mysql_only-buf-pool-size/mysql_only-buf-pool-size.csv outputs/correlation/mysql_datasize/mysql_datasize.csv outputs/correlation/mysql/mysql.csv outputs/correlation/mysql_remove-buf-pool-size/mysql_remove-buf-pool-size.csv outputs/correlation/mysql_rpl-64table-1Mrec-4client/mysql_rpl-64table-1Mrec-4client.csv outputs/correlation/mysql_rpl_select-features/mysql_rpl_select-features.csv outputs/cv/multid_reg_default/multid_reg_default.csv outputs/cv/multid_reg_default/multid_reg_default.pdf outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.csv outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.pdf outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.csv outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.pdf outputs/cv/regression_optimal/regression_optimal.csv outputs/cv/regression_optimal/regression_optimal.pdf outputs/transfer_learning/toy/toy.csv outputs/transfer_learning/toy/toy.pdf outputs/transfer_learning/mysql/mysql.csv outputs/transfer_learning/mysql/mysql.pdf outputs/transfer_learning/macbook/macbook.csv outputs/transfer_learning/macbook/macbook.pdf outputs/transfer_learning/proposed/proposed.csv outputs/transfer_learning/proposed/proposed.pdf outputs/transfer_learning/below_threshold/below_threshold.csv outputs/transfer_learning/below_threshold/below_threshold.pdf outputs/transfer_learning/vanilla/vanilla.csv outputs/transfer_learning/vanilla/vanilla.pdf outputs/transfer_learning/macbook_bias/macbook_bias.csv outputs/transfer_learning/macbook_bias/macbook_bias.pdf outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.csv outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.pdf outputs/parameter_space/fig_proposed/fig_proposed.csv outputs/parameter_space/fig_proposed/fig_proposed.pdf

all_cv: outputs/cv/multid_reg_default/multid_reg_default.csv outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.csv outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.csv outputs/cv/regression_optimal/regression_optimal.csv

all_cv_plot: outputs/cv/multid_reg_default/multid_reg_default.pdf outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.pdf outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.pdf outputs/cv/regression_optimal/regression_optimal.pdf

all_parameter_space: outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.csv outputs/parameter_space/fig_proposed/fig_proposed.csv

all_parameter_space_plot: outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.pdf outputs/parameter_space/fig_proposed/fig_proposed.pdf

all_transfer_learning: outputs/transfer_learning/toy/toy.csv outputs/transfer_learning/mysql/mysql.csv outputs/transfer_learning/macbook/macbook.csv outputs/transfer_learning/proposed/proposed.csv outputs/transfer_learning/below_threshold/below_threshold.csv outputs/transfer_learning/vanilla/vanilla.csv outputs/transfer_learning/macbook_bias/macbook_bias.csv

all_transfer_learning_plot: outputs/transfer_learning/toy/toy.pdf outputs/transfer_learning/mysql/mysql.pdf outputs/transfer_learning/macbook/macbook.pdf outputs/transfer_learning/proposed/proposed.pdf outputs/transfer_learning/below_threshold/below_threshold.pdf outputs/transfer_learning/vanilla/vanilla.pdf outputs/transfer_learning/macbook_bias/macbook_bias.pdf

all_real_data: outputs/real_data/mysql_rpl/mysql_rpl.csv outputs/real_data/mysql/mysql.csv

all_correlation: outputs/correlation/mysql_only-buf-pool-size/mysql_only-buf-pool-size.csv outputs/correlation/mysql_datasize/mysql_datasize.csv outputs/correlation/mysql/mysql.csv outputs/correlation/mysql_remove-buf-pool-size/mysql_remove-buf-pool-size.csv outputs/correlation/mysql_rpl-64table-1Mrec-4client/mysql_rpl-64table-1Mrec-4client.csv outputs/correlation/mysql_rpl_select-features/mysql_rpl_select-features.csv

all_important_parameter: outputs/important_parameter/mysql_rpl/mysql_rpl.csv

outputs/important_parameter/mysql_rpl/mysql_rpl.csv: scripts/conf/important_parameter/mysql_rpl.yaml scripts/important_parameter.py
	python scripts/important_parameter.py --config-name mysql_rpl.yaml

outputs/real_data/mysql_rpl/mysql_rpl.csv: scripts/conf/real_data/mysql_rpl.yaml scripts/real_data.py
	python scripts/real_data.py --config-name mysql_rpl.yaml

outputs/real_data/mysql/mysql.csv: scripts/conf/real_data/mysql.yaml scripts/real_data.py
	python scripts/real_data.py --config-name mysql.yaml

outputs/correlation/mysql_only-buf-pool-size/mysql_only-buf-pool-size.csv: scripts/conf/correlation/mysql_only-buf-pool-size.yaml scripts/correlation.py
	python scripts/correlation.py --config-name mysql_only-buf-pool-size.yaml

outputs/correlation/mysql_datasize/mysql_datasize.csv: scripts/conf/correlation/mysql_datasize.yaml scripts/correlation.py
	python scripts/correlation.py --config-name mysql_datasize.yaml

outputs/correlation/mysql/mysql.csv: scripts/conf/correlation/mysql.yaml scripts/correlation.py
	python scripts/correlation.py --config-name mysql.yaml

outputs/correlation/mysql_remove-buf-pool-size/mysql_remove-buf-pool-size.csv: scripts/conf/correlation/mysql_remove-buf-pool-size.yaml scripts/correlation.py
	python scripts/correlation.py --config-name mysql_remove-buf-pool-size.yaml

outputs/correlation/mysql_rpl-64table-1Mrec-4client/mysql_rpl-64table-1Mrec-4client.csv: scripts/conf/correlation/mysql_rpl-64table-1Mrec-4client.yaml scripts/correlation.py
	python scripts/correlation.py --config-name mysql_rpl-64table-1Mrec-4client.yaml

outputs/correlation/mysql_rpl_select-features/mysql_rpl_select-features.csv: scripts/conf/correlation/mysql_rpl_select-features.yaml scripts/correlation.py
	python scripts/correlation.py --config-name mysql_rpl_select-features.yaml

outputs/cv/multid_reg_default/multid_reg_default.csv: scripts/conf/cv/multid_reg_default.yaml scripts/cross_validation.py
	python scripts/cross_validation.py --config-name multid_reg_default.yaml

outputs/cv/multid_reg_default/multid_reg_default.pdf: outputs/cv/multid_reg_default/multid_reg_default.csv scripts/plot.py scripts/cross_validation.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/cv/multid_reg_default/multid_reg_default.csv ++plot_type=cv ++output_path=outputs/cv/multid_reg_default/multid_reg_default.pdf

outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.csv: scripts/conf/cv/regresson_no-normalize_no-feature-selector.yaml scripts/cross_validation.py
	python scripts/cross_validation.py --config-name regresson_no-normalize_no-feature-selector.yaml

outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.pdf: outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.csv scripts/plot.py scripts/cross_validation.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.csv ++plot_type=cv ++output_path=outputs/cv/regresson_no-normalize_no-feature-selector/regresson_no-normalize_no-feature-selector.pdf

outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.csv: scripts/conf/cv/regression_normalize_feature-selection.yaml scripts/cross_validation.py
	python scripts/cross_validation.py --config-name regression_normalize_feature-selection.yaml

outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.pdf: outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.csv scripts/plot.py scripts/cross_validation.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.csv ++plot_type=cv ++output_path=outputs/cv/regression_normalize_feature-selection/regression_normalize_feature-selection.pdf

outputs/cv/regression_optimal/regression_optimal.csv: scripts/conf/cv/regression_optimal.yaml scripts/cross_validation.py
	python scripts/cross_validation.py --config-name regression_optimal.yaml

outputs/cv/regression_optimal/regression_optimal.pdf: outputs/cv/regression_optimal/regression_optimal.csv scripts/plot.py scripts/cross_validation.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/cv/regression_optimal/regression_optimal.csv ++plot_type=cv ++output_path=outputs/cv/regression_optimal/regression_optimal.pdf

outputs/transfer_learning/toy/toy.csv: scripts/conf/transfer_learning/toy.yaml scripts/transfer_learning.py
	python scripts/transfer_learning.py --config-name toy.yaml

outputs/transfer_learning/toy/toy.pdf: outputs/transfer_learning/toy/toy.csv scripts/plot.py scripts/transfer_learning.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/transfer_learning/toy/toy.csv ++plot_type=transfer_learning ++output_path=outputs/transfer_learning/toy/toy.pdf

outputs/transfer_learning/mysql/mysql.csv: scripts/conf/transfer_learning/mysql.yaml scripts/transfer_learning.py
	python scripts/transfer_learning.py --config-name mysql.yaml

outputs/transfer_learning/mysql/mysql.pdf: outputs/transfer_learning/mysql/mysql.csv scripts/plot.py scripts/transfer_learning.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/transfer_learning/mysql/mysql.csv ++plot_type=transfer_learning ++output_path=outputs/transfer_learning/mysql/mysql.pdf

outputs/transfer_learning/macbook/macbook.csv: scripts/conf/transfer_learning/macbook.yaml scripts/transfer_learning.py
	python scripts/transfer_learning.py --config-name macbook.yaml

outputs/transfer_learning/macbook/macbook.pdf: outputs/transfer_learning/macbook/macbook.csv scripts/plot.py scripts/transfer_learning.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/transfer_learning/macbook/macbook.csv ++plot_type=transfer_learning ++output_path=outputs/transfer_learning/macbook/macbook.pdf

outputs/transfer_learning/proposed/proposed.csv: scripts/conf/transfer_learning/proposed.yaml scripts/transfer_learning.py
	python scripts/transfer_learning.py --config-name proposed.yaml

outputs/transfer_learning/proposed/proposed.pdf: outputs/transfer_learning/proposed/proposed.csv scripts/plot.py scripts/transfer_learning.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/transfer_learning/proposed/proposed.csv ++plot_type=transfer_learning ++output_path=outputs/transfer_learning/proposed/proposed.pdf

outputs/transfer_learning/below_threshold/below_threshold.csv: scripts/conf/transfer_learning/below_threshold.yaml scripts/transfer_learning.py
	python scripts/transfer_learning.py --config-name below_threshold.yaml

outputs/transfer_learning/below_threshold/below_threshold.pdf: outputs/transfer_learning/below_threshold/below_threshold.csv scripts/plot.py scripts/transfer_learning.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/transfer_learning/below_threshold/below_threshold.csv ++plot_type=transfer_learning ++output_path=outputs/transfer_learning/below_threshold/below_threshold.pdf

outputs/transfer_learning/vanilla/vanilla.csv: scripts/conf/transfer_learning/vanilla.yaml scripts/transfer_learning.py
	python scripts/transfer_learning.py --config-name vanilla.yaml

outputs/transfer_learning/vanilla/vanilla.pdf: outputs/transfer_learning/vanilla/vanilla.csv scripts/plot.py scripts/transfer_learning.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/transfer_learning/vanilla/vanilla.csv ++plot_type=transfer_learning ++output_path=outputs/transfer_learning/vanilla/vanilla.pdf

outputs/transfer_learning/macbook_bias/macbook_bias.csv: scripts/conf/transfer_learning/macbook_bias.yaml scripts/transfer_learning.py
	python scripts/transfer_learning.py --config-name macbook_bias.yaml

outputs/transfer_learning/macbook_bias/macbook_bias.pdf: outputs/transfer_learning/macbook_bias/macbook_bias.csv scripts/plot.py scripts/transfer_learning.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/transfer_learning/macbook_bias/macbook_bias.csv ++plot_type=transfer_learning ++output_path=outputs/transfer_learning/macbook_bias/macbook_bias.pdf

outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.csv: scripts/conf/parameter_space/l2s_data_reuse.yaml scripts/parameter_space.py
	python scripts/parameter_space.py --config-name l2s_data_reuse.yaml

outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.pdf: outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.csv scripts/plot.py scripts/parameter_space.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.csv ++plot_type=parameter_space ++output_path=outputs/parameter_space/l2s_data_reuse/l2s_data_reuse.pdf

outputs/parameter_space/fig_proposed/fig_proposed.csv: scripts/conf/parameter_space/fig_proposed.yaml scripts/parameter_space.py
	python scripts/parameter_space.py --config-name fig_proposed.yaml

outputs/parameter_space/fig_proposed/fig_proposed.pdf: outputs/parameter_space/fig_proposed/fig_proposed.csv scripts/plot.py scripts/parameter_space.py scripts/conf/plot.yaml
	python scripts/plot.py ++csv_path=outputs/parameter_space/fig_proposed/fig_proposed.csv ++plot_type=parameter_space ++output_path=outputs/parameter_space/fig_proposed/fig_proposed.pdf

clean:
	rm -rf outputs/*