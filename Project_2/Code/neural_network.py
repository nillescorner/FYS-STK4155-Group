"""
This file contain the neural network function used for project 2.

Feed forward, backward propagation, predict
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp

