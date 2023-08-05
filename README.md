# Learning to Optimize Differentiable Games

Code for ICML 2023 paper: [Learning to Optimize Differentiable Games](). 

## Overview

Many machine learning problems can be abstracted in solving game theory formulations and boil down to optimizing nested objectives, such as generative adversarial networks (GANs) and multi-agent reinforcement learning. 
Solving these games requires finding their stable fixed points or Nash equilibrium. However, existing algorithms for solving games suffer from empirical instability, hence demanding heavy ad-hoc tuning in practice.  
To tackle these challenges, we resort to the emerging scheme of $\textit{Learning to Optimize}$ (L2O), which discovers problem-specific efficient optimization algorithms through data-driven training. Our customized L2O framework for differentiable game theory problems, dubbed $\textit{``Learning to Play Games"}$ (L2PG), seeks a stable fixed point solution, by predicting the fast update direction from the past trajectory, with a novel gradient stability-aware, sign-based loss function. We further incorporate curriculum learning and self-learning to strengthen the empirical training stability and generalization of L2PG. On test problems including quadratic games and GANs, L2PG can substantially accelerate the convergence, and demonstrates a remarkably more stable trajectory.

## Experiments

### Meta-Testing

#### Two-player games (stable)

```bash
CUDA_VISIBLE_DEVICES=0 python sga_l2o_batch.py --eval_game_list stable_game_list_normal.txt --checkpoint checkpoints/two_player.pkl --n_hidden 32 --formula grad,A,S --feat_level o,m0.5,m0.9,m0.99,mt,gt
```

#### Four-player games 

CUDA_VISIBLE_DEVICES=0 python sga_l2o_batch_four_player.py  --checkpoint checkpoints/four_player.pkl  --n_hidden 32 --formula grad,S,A --feat_level o,m0.5,m0.9,m0.99,mt,gt --n_player 4
