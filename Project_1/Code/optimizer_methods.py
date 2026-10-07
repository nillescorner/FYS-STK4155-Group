"""Optimiser methods built on optax (plain, momentum, adagrad, rmsprop, adam)"""

import optax
import jax.numpy as jnp
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradient_descent_methods import cost


def optimiser_solver(method, learning_rate, beta=0.9, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Builds an optax optimizer object for the chosen method.
 
    Based on optax's built-in implementations of SGD, AdaGrad, RMSprop and Adam.
 
        Params:
            method (str): "plain", "momentum", "adagrad", "rmsprop" or "adam"
            learning_rate (float): learning rate used by the optimizer
            beta (float): momentum coefficient, only used for "momentum", default: 0.9
            rho (float): decay rate for the squared-gradient average, only used for "rmsprop", default: 0.99
            beta1 (float): first-moment decay rate, only used for "adam", default: 0.9
            beta2 (float): second-moment decay rate, only used for "adam", default: 0.999
            eps (float): small constant added for numerical stability, default: 1e-8
 
        Returns:
            optimizer (optax.GradientTransformation): the configured optax optimizer object
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

def optimiser(method, theta, g, state, t, gamma, beta=0.9, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Performs one manual (non-optax) update of theta from the gradient g at step t (t = 1, 2, ...).
 
    Based on Eqs. (4.10), (4.28), (4.42)-(4.45), (4.47)-(4.48) and (4.51)-(4.55) from the lecturebook.
 
        Params:
            method (str): "plain", "momentum", "adagrad", "rmsprop" or "adam"
            theta (NDArray): current parameter vector
            g (NDArray): gradient of the cost function at theta
            state (dict): running quantities (e.g. "v", "r", "m") carried between steps
            t (int): current step number, starting at 1
            gamma (float): learning rate
            beta (float): momentum coefficient, only used for "momentum", default: 0.9
            rho (float): decay rate for the squared-gradient average, only used for "rmsprop", default: 0.99
            beta1 (float): first-moment decay rate, only used for "adam", default: 0.9
            beta2 (float): second-moment decay rate, only used for "adam", default: 0.999
            eps (float): small constant added for numerical stability, default: 1e-8
 
        Returns:
            theta (NDArray): updated parameter vector
            state (dict): updated running quantities
    """

    if method == "plain":
        return theta - gamma * g, state
    if method == "momentum":
        v = beta * state.get("v", 0.0) + gamma * g               # Eq. (4.28)
        state["v"] = v
        return theta - v, state
    if method == "adagrad":
        r = state.get("r", 0.0) + g * g                           # Eq. (4.42)
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.45)
    if method == "rmsprop":
        r = rho * state.get("r", 0.0) + (1.0 - rho) * g * g       # Eq. (4.47)
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.48)
    if method == "adam":
        m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g       # Eq. (4.51)
        r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g   # Eq. (4.52)
        state["m"], state["r"] = m, r
        m_hat = m / (1.0 - beta1**t)                              # Eq. (4.54)
        r_hat = r / (1.0 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state   # Eq. (4.55)
    raise ValueError(f"unknown method {method}")

def optimiser_step(optimizer, theta, g, opt_state):
    """
    Performs one optax-based update of theta from the gradient g.
 
        Params:
            optimizer (optax.GradientTransformation): optax optimizer object, from optimiser_solver
            theta (NDArray): current parameter vector
            g (NDArray): gradient of the cost function at theta
            opt_state (optax state): optax optimizer state carried between steps
 
        Returns:
            theta (NDArray): updated parameter vector
            opt_state (optax state): updated optax optimizer state
    """

    updates, opt_state = optimizer.update(g, opt_state, theta)
    theta = optax.apply_updates(theta, updates)
    return theta, opt_state


def optimise(grad, optimizer, num_iters):
    # LLM assisted
    """
    JIT-compiled fixed-length Optax optimization routine.

        Params:
            grad (callable): gradient function of the objective to be minimized.
            optimizer (optax optimizer): optimizer state and update rule.
            num_iters (int): number of optimization steps to perform.

        Returns:
            theta (array-like): optimized parameter vector after the specified number of iterations.
    """

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


def funct_comparison(X, y, FUNCT_RUNS, num_iters=1000, lmbda=0.0):
    # LLM assisted
    """
    Compare optimizer performance using excess cost based on a linear regression objective.

        Params:
            X (array-like): design matrix containing the input features.
            y (array-like): target values for the regression problem.
            FUNCT_RUNS (int): number of optimizer runs used for comparison.
            num_iters (int): maximum number of iterations for each optimization routine.
            lmbda (float): regularization parameter used in the cost function.

        Returns:
            comparison (dict): dictionary containing the optimizer comparison results,
                including excess cost values, convergence traces, and final parameter estimates.
    """

    Xj = jnp.array(X)
    yj = jnp.array(y)

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
