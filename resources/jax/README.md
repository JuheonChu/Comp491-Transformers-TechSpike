This document puts together our learning of JAX.

## JAX basics for machine learning

Most tutorials can be found on [JAX 101 documentation](https://jax.readthedocs.io/en/latest/jax-101/index.html) and [AISummer](https://theaisummer.com/jax/) article.


Below are they key JAX features/transforms used in machine learning. 

1) JIT (Just-in-time compilation)
2) Autograd (Auto differentiation with grad)
3) Auto vectorization with vmap()
4) Auto parallelization with pmap()
5) Pseudo-random number generator (PRNG)

**1) JIT (Just-in-time compilation)**

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

Facts
- Use pure functions (otherwise untracked side effects)
- Uses a tracer object under the hood

**2) Autograd (Auto differentiation with grad)**
 
Calculate gradients (derivatives), including 2nd/3rd order gradients 

**3) Auto vectorization with vmap()**

...still trying to understand...

**4) Auto parallelization with pmap()**

Distribute multiple computations across TPU/GPU for a higher performance

Distribute 

## Other useful information
- JAX arrays are immutable 
- Here is a helpful [gotcha list](https://github.com/google/jax#current-gotchas)

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
| Vectorization | |
| Parallelization | |
