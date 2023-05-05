# DroQ
The [DroQ](https://openreview.net/pdf?id=xCVJMsPv3RT) algorithm is a recent Q-learning algorithm developed as a sample efficient version of [REDQ](https://arxiv.org/abs/2101.05982). 

All algorithm's step are very similar to SAC, except for the fact that there are many Q networks, and that the target Q value is computed using the minimum over a randomly selected batch of Q networks' values. 