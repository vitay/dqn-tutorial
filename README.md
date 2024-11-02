# DQN agent on CartPole using pytorch

Basic implementation of DQN in pytorch, applied to the CartPole environment.

## Installation

The python dependencies are managed using [uv](https://github.com/astral-sh/uv). After cloning the repo, simply type:

```bash
uv sync
```

Otherwise, `pip install -e .` should install everything you need in your virtual environment:

* gymnasium>=1.0.0
* matplotlib>=3.9.2
* numpy>=2.1.2
* pyyaml>=6.0.2
* torch>=2.5.1
* wandb>=0.18.5

## Usage

Everything is in the `train_DQN.py` script. 

```python
python train_DQN.py
```

The hyperparameters are in the `config.yaml` file.

Progress is tracked using Weights & Biases <https://wandb.ai/>.