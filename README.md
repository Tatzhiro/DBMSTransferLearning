# DBMS Transfer Learning

## Project Structure

* scripts: Runnable scripts to compare transfer learning methods
* regression: Internal Transfer Learning modules
* dataset: DBMS performance data used for experiments
* neural_network: Cache folder that holds MLP weights and others

## How to run

There are three experiments that can be run: `{context_retrieval, full_transfer, data_transfer}`.

Run the following command with the experiment option of your choice.

The outputs will be saved in `outputs/transfer_learning` directory

```shell
$ PYTHONPATH=. python scripts/batch.py ${EXPERIMENT_OPTION}
Running full transfer…
```
