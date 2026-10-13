"""
This code contains the optimizer methods we will use in project 2.

Gradient descent, stochastich gradient descent, RMSprop, adam.
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp

