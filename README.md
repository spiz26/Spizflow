# Spizflow
Machine learning, Deep learning implementation  
It's a code that I implemented myself.  
Spizflow the latest ver.2.5

1. Layer : Dense, Dropout, Convolution, Fully Connected, pooling
2. Optimizer : SGD, Momentum, AdaGrad, RMSProp, Adam
3. Activation function : Linear, Sigmoid, tanh, Softmax, ReLU, leakey relu, ELU, PReLU
4. Model Class
5. Initialization : Xavier, He
6. Loss visualization

ver.1.0 : Basic deep learning code implement.  
ver.1.1 : I don't remember. Perhaps there were only SGD, sigmoid.  
ver.1.2 : Adding Adagrad, ReLU.  
ver.1.3 : Adding dropout layer.  

ver.2.0 : Organize the code neatly using the class. POP -> OOP.  
ver.2.1 : Adding mini batch, some Optimizers.   
ver.2.2 : CNN update. Adding Convolution, Fully connected, pooling layers.  
ver.2.3 : Adding some Activation functions(Linear, tanh, leaky relu, ELU, PReLU).  
ver.2.4 : Adding Initialization.  
ver.2.5 : Adding Loss visualization.  
ver.2.6 : Adding Numerical differentiation and using zero bias initialization

ver.2.3 accuracy

```
[0/10 epochs] Train Accuracy : 9.925%, Test Accuracy : 11.6%
[1/10 epochs] Train Accuracy : 83.325%, Test Accuracy : 79.2%
[2/10 epochs] Train Accuracy : 92.1%, Test Accuracy : 86.4%
[3/10 epochs] Train Accuracy : 92.0%, Test Accuracy : 87.6%
[4/10 epochs] Train Accuracy : 94.575%, Test Accuracy : 90.3%
[5/10 epochs] Train Accuracy : 95.225%, Test Accuracy : 90.1%
[6/10 epochs] Train Accuracy : 94.925%, Test Accuracy : 91.1%
[7/10 epochs] Train Accuracy : 97.075%, Test Accuracy : 92.1%
[8/10 epochs] Train Accuracy : 96.9%, Test Accuracy : 92.3%
[9/10 epochs] Train Accuracy : 97.05%, Test Accuracy : 91.7%
[10/10 epochs] Train Accuracy : 96.75%, Test Accuracy : 91.6%
Training complete! 2.0minutes, 57.55seconds
```

ver.2.4 accuracy

```
[0/10 epochs] Train Accuracy : 9.95%, Test Accuracy : 11.5%
[1/10 epochs] Train Accuracy : 92.425%, Test Accuracy : 88.2%
[2/10 epochs] Train Accuracy : 95.3%, Test Accuracy : 91.5%
[3/10 epochs] Train Accuracy : 95.75%, Test Accuracy : 92.2%
[4/10 epochs] Train Accuracy : 94.15%, Test Accuracy : 89.4%
[5/10 epochs] Train Accuracy : 97.85%, Test Accuracy : 94.6%
[6/10 epochs] Train Accuracy : 97.1%, Test Accuracy : 93.4%
[7/10 epochs] Train Accuracy : 98.675%, Test Accuracy : 95.3%
[8/10 epochs] Train Accuracy : 98.4%, Test Accuracy : 95.1%
[9/10 epochs] Train Accuracy : 98.425%, Test Accuracy : 93.0%
[10/10 epochs] Train Accuracy : 98.725%, Test Accuracy : 94.4%
Training complete! 2.0minutes, 55.77seconds
```

ver.2.6 accuracy

```
[0/10 epochs] Train Accuracy : 9.75%, Test Accuracy : 10.4%
[1/10 epochs] Train Accuracy : 93.45%, Test Accuracy : 89.8%
[2/10 epochs] Train Accuracy : 96.975%, Test Accuracy : 92.5%
[3/10 epochs] Train Accuracy : 97.95%, Test Accuracy : 93.5%
[4/10 epochs] Train Accuracy : 97.575%, Test Accuracy : 92.4%
[5/10 epochs] Train Accuracy : 98.075%, Test Accuracy : 93.3%
[6/10 epochs] Train Accuracy : 96.975%, Test Accuracy : 91.3%
[7/10 epochs] Train Accuracy : 98.275%, Test Accuracy : 93.3%
[8/10 epochs] Train Accuracy : 98.775%, Test Accuracy : 93.5%
[9/10 epochs] Train Accuracy : 99.125%, Test Accuracy : 93.8%
[10/10 epochs] Train Accuracy : 99.0%, Test Accuracy : 94.3%
Training complete! 1.0minutes, 28.54seconds
```
