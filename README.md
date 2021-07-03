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
|`Python`|  `3.7`|
|`torch`|  `1.9.0`|
|`selenium`|  `3.141.0`|
|`Pillow`|  `8.3.0`|
|`torchvision`|  `0.10.0`|
|`matploitlib`|  `3.4.2`|

## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`DeepDino.py`| general purpose main application for training/playing a LS-DQN agent|
|`dqn_model.py`| main application tailored for Atari's Pong|
|`game.py`| main application tailored for Atari's Boxing|
|`game_state.py`| sample code for playing a game, also in `ls_dqn_main.py`|
|`config.py`| classes for actions selection (argmax, epsilon greedy)|
|`dino_agent.py`| agent class, holds the network, action selector and current state|
|`dqn_model.py`| DQN classes, neural networks structures|
|`*.pth`| Checkpoint files for the Agents (playing/continual learning)|

## References
* 
* Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou,
Daan Wierstra, Martin Riedmiller [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), NIPS 2013