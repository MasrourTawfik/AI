# DRL Heating Process — Open Reproducibility Companion

This repository companion implements a **transparent, synthetic, conceptual reproduction** of the workflow described in:

> C. El Mazgualdi, T. Masrour, I. El Hassani, A. Khdoudi, “A Deep Reinforcement Learning (DRL) Decision Model for Heating Process Parameters Identification in Automotive Glass Manufacturing,” *Artificial Intelligence and Industrial Applications: Smart Operation Management*, pp. 77–87, Springer, 2021. DOI: 10.1007/978-3-030-51186-9_6.

## What this companion reproduces

The paper describes an offline decision system that combines:

1. a **self-prediction artificial neural network** for the furnace/process response; and
2. a **Deep Q-Network (DQN)** that selects the main heating-process parameters, specifically glass transfer speed and zone temperature, from a desired outlet glass temperature.

This open companion implements the same high-level decision architecture on a **synthetic transparent process model** so that the complete workflow can be executed, inspected and extended without access to proprietary industrial data.

## Important scientific-status note

This is **not the original author code** and it does **not reproduce the confidential/industrial furnace dataset**. It is intentionally labelled a *conceptual reproduction companion*. Numerical values produced by this repository must not be reported as the paper's original experimental results.

## Pipeline

`synthetic furnace data -> ANN surrogate -> DQN controller -> target outlet temperature -> speed / zone-temperature recipe`

The synthetic process is constructed so that transfer speed and zone temperature have a plausible nonlinear influence on outlet temperature. The ANN learns that transparent data generator, and the DQN then learns to adjust the two process parameters until the predicted outlet temperature falls within a specified tolerance.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python drl_heating_repro.py
```

For a shorter smoke run:

```bash
python drl_heating_repro.py --episodes 120
```

Outputs are written to `artifacts/`:

- `surrogate_ann.pt`
- `dqn_agent.pt`
- `metrics.json`

## Reproducibility guarantees in this companion

- deterministic random seeds;
- fully synthetic data generator included in source;
- explicit process bounds and action discretization;
- ANN-surrogate validation metrics;
- DQN training and fixed target-policy evaluation;
- no hidden external service or proprietary dataset dependency.

## Verified local quality gate

The committed example run was validated before publication:

- unit tests: **2/2 passed**;
- surrogate MAE: about **3.51 °C**;
- surrogate RMSE: about **4.56 °C**;
- last-50-episode success rate: **0.98** at a ±3 °C tolerance;
- fixed-target evaluation succeeded on 5 of 6 targets within ±3 °C, with the remaining target within about 4.2 °C.

See `example_metrics.json` for the complete verified run.

## Colab

Open `DRL_Heating_Open_Reproduction.ipynb` in Google Colab. The notebook can clone this repository automatically when needed.

## Citation

If this companion helps you understand or extend the method, please cite the **original Springer chapter**, not this repository:

```bibtex
@incollection{elmazgualdi2021drlheating,
  title={A Deep Reinforcement Learning (DRL) Decision Model for Heating Process Parameters Identification in Automotive Glass Manufacturing},
  author={El Mazgualdi, Choumicha and Masrour, Tawfik and El Hassani, Ibtissam and Khdoudi, Abdelmoula},
  booktitle={Artificial Intelligence and Industrial Applications: Smart Operation Management},
  pages={77--87},
  year={2021},
  publisher={Springer},
  doi={10.1007/978-3-030-51186-9_6}
}
```

## Research extensions

Natural research directions include continuous-action RL (DDPG/SAC), uncertainty-aware surrogate models, constrained/safe RL, multi-zone furnace models, energy-aware rewards, digital-twin integration and transfer from synthetic to industrial data.
