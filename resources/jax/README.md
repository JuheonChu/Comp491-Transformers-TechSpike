This document puts together our learning of JAX and other information we found interesting.

## JAX basics for machine learning

Most tutorials can be found on [JAX 101 documentation](https://jax.readthedocs.io/en/latest/jax-101/index.html) and [AISummer](https://theaisummer.com/jax/) article.

Below are they key JAX features used in machine learning. 

1) JIT (Just-in-time compilation)
2) Autograd (Auto differentiation with grad)
3) Auto vectorization with vmap()
4) Auto parallelization with pmap()
5) Pseudo-random number generator (PRNG)

**JIT (Just-in-time compilation)**

Can be used as a decorator 
```
@jit
def func(x, y):
  print("Hello transformers")
```

or higher-order function

```
def func(x, y):
  print("Hello transformers")
  
func_2 = jit(slow_f)
```
- Use pure functions 


## Demo: NLP Model with JAX

**Tutorial**

[Getting started with NLP using JAX](https://www.kaggle.com/code/guillemkami/getting-started-with-nlp-using-jax/notebook)

**Platform**

To reduce hardware complexity, we installed jax and most of the support packages for GPU (CONDA, cuDNN) in a shared Google colab.
- CUDA version 11.2
- Python 3.7.15
- 

## Glossary

| Terms  | Definition |
| ------------- | ------------- |
| [CUDA](https://developer.nvidia.com/cuda-downloads)  |  A parallel computing platform and programming model which harnesses GPU power |
| [cuDNN](https://developer.nvidia.com/CUDNN) | Deep Neural Network GPU-accelerated library |
| TPU | Tensor Processing Unit. Hardware accelerator developed by Google for neural network and ML  |
| SPMD | Single-program multiple-data (SPMD) programs |
