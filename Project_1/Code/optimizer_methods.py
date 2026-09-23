"""Optimiser methods built on optax (plain, momentum, adagrad, rmsprop, adam)"""

import optax
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

from gradient_descent_methods import cost


def optimiser_solver(method, learning_rate, beta=0.9, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Using optax to implement the methods
    Args:
        method : "plain", "momentum", "adagrad", "rmsprop", "adam"
        learning_rate : learning rate of each model
        
    beta, rho, beta1 and beta2  are all used track the magnitude of previous gradients
    beta for "momentum"
    rho for "rmsprop"
    beta1 and beta2 for "adam"
        
    """
    
    if method == "plain":
        optimizer = optax.sgd(learning_rate=learning_rate)

    elif method == "momentum":
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=beta)

    elif method == "adagrad":
        optimizer = optax.adagrad(learning_rate=learning_rate, eps=eps)

    elif method == "rmsprop":
        optimizer = optax.rmsprop(learning_rate=learning_rate, decay=rho, eps=eps)

    elif method == "adam":
        optimizer = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2, eps=eps)

    else:
        raise ValueError(f"unknown method {method}")
    return optimizer


def optimiser_step(optimizer, theta, g, opt_state):
    updates, opt_state = optimizer.update(g, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    return theta, opt_state


def optimise(grad, optimizer, num_iters):
    """JIT-compiled fixed-length Optax optimization."""

    @jax.jit # very fast! me like!
    def run(theta0):
        theta0 = jnp.asarray(theta0, dtype=jnp.float64)
        state0 = optimizer.init(theta0)

        def step(carry, _):
            theta, state = carry
            g = grad(theta)

            updates, state = optimizer.update(
                g, state, theta
            )
            theta = optax.apply_updates(theta, updates)

            return (theta, state), theta

        (_, _), path = jax.lax.scan(
            step, (theta0, state0), xs=None,length=num_iters)

        return jnp.concatenate([theta0[None, :], path], axis=0 )

    return run


def funct_comparison(Xj, yj, FUNCT_RUNS, num_iters=1000, lmbda=0.0):
    """Compare optimizers using excess cost."""


    n, p = Xj.shape
    theta_cf = jnp.linalg.solve(
        Xj.T @ Xj + n * lmbda * jnp.eye(p),
        Xj.T @ yj)
    c_min = cost(theta_cf, Xj, yj, lmbda=lmbda)

    theta0 = jnp.zeros(Xj.shape[1])
    grad_cost = jax.grad(cost, argnums=0)
    grad = lambda theta: grad_cost(theta, Xj, yj, lmbda=lmbda)

    # Largest Hessian eigenvalue without explicitly constructing eigs.
    hessian_eigs = jnp.linalg.eigvalsh((2.0 / n) * Xj.T @ Xj + 2.0 * lmbda * jnp.eye(Xj.shape[1]))
    gamma_gd = float(0.9 * 2.0 / jnp.max(hessian_eigs))


    def excess_cost(theta):
        return cost(theta, Xj, yj, lmbda=lmbda) - c_min
    excess_costs = jax.jit(jax.vmap(excess_cost))

    out = {}

    for method, gammas_, color in FUNCT_RUNS:
    
        out[method] = {}

        for gamma in gammas_:
            learning_rate = gamma_gd if gamma is None else gamma

            optimizer = optimiser_solver(method, learning_rate)

            run = optimise(grad, optimizer, num_iters)
            path = run(theta0)
            
            out[method][gamma] =  excess_costs(path) #had to change this so that we avoid converting between jax and np. made it super slow

            if gamma == gammas_[-1]:
                print(f"Runs for {method=} are done")

    return out, hessian_eigs, gamma_gd
