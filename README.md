## Install

First, clone the repository:

```
git clone https://github.com/zhaoyang02/Contrastive-Neural-Process.git
```

Then install the dependencies as listed in `env.yml` and activate the environment:

```
conda env create -f env.yml
conda activate tnp
```

## Usage

### Training
```
python gp.py --mode=train --expid=cl-np --model=cl-np
```
The config of hyperparameters of each model is saved in `configs/gp`. If training for the first time, evaluation data will be generated and saved in `evalsets/gp`. Model weights and logs are saved in `results/gp/{model}/{expid}`.

### Evaluation
```
python gp.py --mode=evaluate_all_metrics --expid=cl-np --model=cl-np
```
Note that you have to specify `{expid}` correctly. The model will load weights from `results/gp/{model}/{expid}` to evaluate.

### Results

All experimental results locate at `results/gp/{model}/{expid}/eval_rbf_all_metrics.txt`

## Acknowledgement

The implementation of the baselines is borrowed from the official code base of [Transformer Neural Processes](https://github.com/tung-nd/TNP-pytorch).
