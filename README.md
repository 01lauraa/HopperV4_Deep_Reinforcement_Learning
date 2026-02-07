# Deep Reinforcement Learning – Hopper & BipedalWalker

This repository contains the code for a Deep Reinforcement Learning assignment focused on continuous control tasks using Hopper-v4 and BipedalWalker-v3 from Gymnasium. The goals are (1) study robustness and generalization under environment changes, and (2) compare on-policy and off-policy methods in terms of performance, stability, and computational cost.
---

## Overview of Experiments

### Hopper-v4: Robustness to Mass Changes

For Hopper-v4, the focus is on generalization under dynamics changes. Models are trained with different torso masses and then evaluated across a range of unseen torso masses.

Key points:
- PPO is trained with fixed torso masses (3, 6, and 9 kg)
- During evaluation, the torso mass is systematically varied
- Performance degradation and robustness are analyzed

This is implemented using a custom environment wrapper that directly modifies the torso mass in the MuJoCo model.

### BipedalWalker-v3: PPO vs TQC

For BipedalWalker-v3, two algorithms are compared:
- PPO (on-policy): simple and stable, but sample-hungry
- TQC (off-policy): more complex and computationally expensive, but highly sample-efficient

PPO hyperparameters are tuned using Optuna, while TQC uses benchmark hyperparameters. Early stopping is explored for TQC to reduce training time while maintaining strong performance.

---

## Algorithms

- PPO (Proximal Policy Optimization)  
  Chosen for its simplicity, stability, and reasonable performance without extensive tuning. A small entropy coefficient is added to encourage exploration.

- TQC (Truncated Quantile Critics)  
  Used for BipedalWalker-v3 due to its strong performance on continuous control benchmarks and its ability to control overestimation bias through distributional value estimation.

---

P1.py                # Hopper-v4 experiments
PPO+OPTUNA.ipynb     # PPO tuning (BipedalWalker-v3)
TQC.ipynb            # TQC training and evaluation
results_completed/   # Saved models and results
report.pdf           # Full report
README.md


---

## Results Summary

- PPO generalizes poorly when test torso mass differs significantly from the training mass.
- TQC outperforms PPO on BipedalWalker-v3 in terms of sample efficiency and final reward.
- Early stopping allows TQC to reach strong performance with far fewer training steps.
- PPO is easier to tune and lighter to run, but less robust and less sample-efficient.

Detailed results and discussion are provided in report.pdf.

---


