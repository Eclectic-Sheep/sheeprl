# DroQ
The [DroQ](https://openreview.net/pdf?id=xCVJMsPv3RT) algorithm is a recent Q-learning algorithm developed as a sample efficient version of [REDQ](https://arxiv.org/abs/2101.05982). 

All algorithm's step are very similar to SAC, except for the fact that
  * There are M Q-networks and M Q-target networks
  * Q-networks and Q-target networks have *dropout* layers
  * For the value loss, the target Q value is computed using the minimum over the M Q-target networks' values (line 6 of Algorithm 2 in the paper)
  * For the policy loss, the Q value is computed using the mean over the M Q networks' values (line 10 of Algorithm 2 in the paper)