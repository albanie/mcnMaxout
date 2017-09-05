# Maxout Networks

This directory contains an implementation of the maxout unit and provides 
a simple classification example on CIFAR-10.  It is based on code by 
Jia-Ren Chang, which can be found 
[here](https://github.com/JiaRenChang/Batch_Normalized_Maxout_NIN).

### Installation

This module can be installed with the MatConvNet `vl_contrib` package manger:

```
vl_contrib('install', 'mcnMaxout') ;
vl_contrib('compile', 'mcnMaxout') ;
vl_contrib('setup', 'mcnMaxout') ;
vl_contrib('test', 'mcnMaxout') ; % optional
```

Compilation is worthwhile, since the compiled `vl_nnmaxout` CUDA function is at least an order of magnitude faster than the naive MATLAB equivalent (currently `vl_nnmaxout` supports only GPU, and not CPU computation).  However, if speed is not an issue, a pure MATLAB implementation is also included (as `vl_nnmaxout_matlab`), which can be run on either the CPU or GPU.   

### Demo

Running the `cnn_cifar_maxout.m` script will download a copy of CIFAR-10 and train a simple maxout network for classification.  As a reference model, this should reach approximately 92% accuracy.

### Notes

Maxout was originally introduced and studied in the [paper](https://arxiv.org/pdf/1302.4389.pdf):

  `Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y.`         
  `(2013). Maxout networks. arXiv preprint arXiv:1302.4389.`

It has seen a wide range of applications in different domains and seems to generalise well.  For instance, it can be applied directly to the Network-in-Network architecture, as shown 
in the [this work](https://arxiv.org/abs/1511.02583):

  `Chang, J. R., & Chen, Y. S. (2015). Batch-normalized maxout network in network.`   
  `arXiv preprint arXiv:1511.02583.`

It can also be used as an efficient strategy for network pruning (compression), as described
in the [here](https://arxiv.org/pdf/1707.06838.pdf):

`Rueda, F. M., Grzeszick, R., & Fink, G. A. (2017). Neuron Pruning for Compressing`   
`Deep Networks using Maxout Architectures. arXiv preprint arXiv:1707.06838.`
