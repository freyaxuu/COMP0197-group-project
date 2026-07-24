# COMP0197 Group Project

Short description: this repository contains the COMP0197 applied deep learning group project for GBP/USD exchange-rate forecasting. It includes data preparation, feature selection, LSTM-based forecasting models, Gaussian uncertainty-aware variants, training scripts, evaluation utilities, and saved model checkpoints.

## Environment

The project was prepared with the local micromamba environment:

```bash
/Users/apple/micromamba/envs/comp0197-pt
```

To recreate the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run training:

```bash
python train.py
```

Run evaluation:

```bash
python test.py
```

The scripts expect or create data under `data/` and output figures/results during training and testing. Saved `.pt` checkpoints for the trained models are included in the repository.
