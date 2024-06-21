# CIFAR10
These are entry level projects to help me understand machine learning.

Deep neural network with GELU activation function with fast convergence. 

This particual project is focused on MNIST:

http://yann.lecun.com/exdb/mnist/

The architecture is as follows:
Input: 28x28
HiddenLayer1: Linear + GELU (in: 200, out: 200)
HiddenLayer2: Linear + GELU (in: 200, out: 200)
Output: Linear + GELU (200,10)

Loss function: SoftMmax + Cross Entropy Loss
Optimizer: ADAM

99% Accuracy.
