## Project 1 - FYS-STK4155

Hello and welcome to the Project 1 folder in our FYS-STK4155 repository for the group consisting of Pernille Xu Amundsen, Maria Andersen and Hannah Westgaard. 

In this project we will explore the Runge functions:

$$f(x) = \frac{1}{1 + 25x^2}, \quad x \in [-1, 1]$$

with added Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, and study:


- The linear regression methods: Ordinary Least Squares (OLS), Ridge and Lasso. 
- The resampling techniques: Bootstrap (to studi bias-variance tradeoff) and cross-validation (using k-folding)
- Gradient descent methods with both analytical and automatic differentiation using JAX.
- Gradient descent by including momentum, AdaGrad, RMSprop and Adam for iteratively updating the learning rate
- Stochastic gradient descent

The project is structured as:

---
```
Project_1/
├── Code/
│   ├── _Part_a.py                  # OLS regression analysis
│   ├── _Part_b.py                  # Ridge regression analysis
│   ├── _Part_c.py                  # Bias-variance via bootstrap
│   ├── _Part_d.py                  # k-fold CV (k = 5, 10)
│   ├── _Part_e.py                  # Plain GD, analytical vs JAX 
│   ├── _Part_f.py                  # Momentum, AdaGrad, RMSprop, Adam (optax)
│   ├── _Part_g.py                  # Lasso regression
│   ├── _Part_h.py                  # Stochastic gradient descent, batch size and learning rate
│   ├── _Part_i.py                  # Final model selection: OLS, Ridge, Lasso with CV
│   │
│   ├── general_functions.py        # Runge function, data generation, design matrix, scaling, FIG_DIR
│   ├── regression_methods.py       # Closed-form OLS/Ridge, MSE and R²
│   ├── resampling_methods.py       # Bootstrap, cross-validation, grid search
│   ├── gradient_descent_methods.py # Cost, gradients, Hessian eigenvalues, plain GD, closed form
│   ├── optimizer_methods.py        # Update rules and optax-based optimiser comparison
│   ├── sgd_methods.py              # Mini-batches, learning-rate schedule, SGD
│   ├── lasso_methods.py            # Soft thresholding, Lasso GD, coordinate descent
│   ├── plots.py                    # Heatmap and coefficient plotting helpers
│   └── colors.py                   # Color scheme throughout project
├── figs/                           # Generated figures used in Report
├── report/                         # Finalized Report (PDF)
├── requirements.txt                # Requirements to run this program
└── README.md
```
---

#### Installation
Python 3.10 or newer is recommended. Our group used a mixture of 3.11.9 and 3.12.7. Install the dependencies via:

pip install -r requirements.txt 

---

#### Running the code
All scripts import functions, variables etc from the same folder, so run them from inside Code by

```bash
cd Code
python _Part_x.py
```

Each _Part_x-py file corresponds to the matching part of the project description and can be run independantly.

## Use of LLMs

See the declaration at the end of the report. Code sections written with LLM assistance of level 2 or above are marked with comments (`# LLM Assisted`) in the source files.