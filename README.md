#Deep Dino Run
No internet? No Problem! We will teach a deep neural network to play Chrome's Dino Game!
A pytorch implementation..

Based on the paper:

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou,
Daan Wierstra, Martin Riedmiller [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), NIPS 2013

![dino run](https://github.com/Gal-Kinberg/DeepDinoRun/blob/revert_checks/dino_run.gif)


- [Deep Dino Run](#Deep Dino Run)
  * [Background](#background)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Playing Atari on Windows](#playing-atari-on-windows)
  * [References](#references)

## Background
The idea of this algorithm i

## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5`|
|`torch`|  `1.9.0+cu102`|
|`selenium`|  `3.141.0`|

## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`ls_dqn_main.py`| general purpose main application for training/playing a LS-DQN agent|
|`pong_ls_dqn.py`| main application tailored for Atari's Pong|
|`boxing_ls_dqn.py`| main application tailored for Atari's Boxing|
|`dqn_play.py`| sample code for playing a game, also in `ls_dqn_main.py`|
|`actions.py`| classes for actions selection (argmax, epsilon greedy)|
|`agent.py`| agent class, holds the network, action selector and current state|
|`dqn_model.py`| DQN classes, neural networks structures|
|`experience.py`| Replay Buffer classes|
|`hyperparameters.py`| hyperparameters for several Atari games, used as a baseline|
|`srl_algorithms.py`| Shallow RL algorithms, LS-UPDATE|
|`utils.py`| utility functions|
|`wrappers.py`| DeepMind's wrappers for the Atari environments|
|`*.pth`| Checkpoint files for the Agents (playing/continual learning)|
|`Deep_RL_Shallow_Updates_for_Deep_Reinforcement_Learning.pdf`| Writeup - theory and results|

## References
* 
* Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou,
Daan Wierstra, Martin Riedmiller [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), NIPS 2013