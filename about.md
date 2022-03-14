---
title: Papers
---

Links to and summaries of recent publications:

[*Deep Phasor Networks*](https://arxiv.org/pdf/2106.11908): Demonstrated deep neural networks which utilize a phasor-based activation function. These networks can compete with traditional networks on common image-recognition tasks, and carry the advantage that they can inherently be executed via a spiking mode without any conversion step between modes. [Code](https://github.com/wilkieolin/phasor_networks)

[*Bridge Networks*](https://dl.acm.org/doi/pdf/10.1145/3477145.3477161): Applied concepts from vector-symbolic architectures to create a network which learns the relationship between two domains of information. E.g. given an image, it can predict a label, and when given a label, it can predict an image. The unique architecture used also allows the network to perform self-distillation and exhibit potential continual learning capabilities. [Code](https://github.com/wilkieolin/bridge_networks)

[*A Dual-Memory Architecture for Reinforcement Learning on Neuromorphic Platforms*](https://iopscience.iop.org/article/10.1088/2634-4386/ac1a64/pdf): Designed and implemented a model for reinforcement learning implemented on the Intel Loihi neuromorphic chip. This model utilized two 'memory' segments - one slow, one fast - to enable spike-based reinforcement learning entirely on-chip. The tasks of the multi-arm bandit, a maze, and blackjack were demonstrated. [Code](https://github.com/wilkieolin/loihi_rl)

[*Stochasticity and Robustness in Spiking Neural Networks*](https://www.sciencedirect.com/science/article/am/pii/S0925231220313035): Investigated the use of stochasticity in training spiking neural networks and its impact on the robustness of these networks to weight perturbation. Demonstrated that networks trained under stochastic conditions are likelier to have improved robustness.

[*Cellular Memristive-Output Reservoir (CMOR)*](https://arxiv.org/pdf/1906.06414): Implemented a cellular-automata based reservoir system in a 14-nm CMOS process. Tested resulting integrated circuits and demonstrated that a memristor-based readout of the reservoir was able to carry out a non-linear classification problem on the inputs (XOR). 