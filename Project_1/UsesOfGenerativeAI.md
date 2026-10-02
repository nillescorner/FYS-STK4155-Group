Hey! Lets use this txt-file to keep track of what we're using generative AI for!

Guide for labeling as provided in MachineLearningUiO:
Level 	Label 	      Meaning
0 	    None 	        No LLM assistance
1 	    Debugging 	  LLM helped identify a bug; fix implemented by you
2 	    Snippet 	    LLM provided a function, loop, or short block; you integrated and tested it
3 	    Skeleton 	    LLM generated the overall structure of a script/class; you filled in domain-specific logic
4 	    Substantial 	LLM wrote the majority of a file; you adapted, tested, and commented it


TEMP

1. Debugging code
2. Clarifying consepts to better understanding
3. Plotting function for regression
4. Understading the regulatization parameter lambda when comparing estimators (GPT-5.6 Terra)
5. Improved reusability of calculated arrays with copilot GPT.6 Luna

**Example**

| File / notebook         | LLM level                                 | Description                              |
| ----------------------- | ----------------------------------------- | ---------------------------------------- |
| train.py                | 2                                         | LLM provided the shot/measurement setup; |
|                         | tested and adapted for our NTK experiment |                                          |
| qaoa.ipynb              | 3                                         | LLM generated class skeleton for QAOA;   |
|                         | Optimization architecture designed by us  |                                          |
| utils/ntk_compute.py    | 0                                         | Written independently                    |
| results/plot_figures.py | 1                                         | LLM debugged an indexing error in the    |
|                         | eigenvalue sorting routine                |                                          |
|                         |                                           |                                          |

**Ours**

| File / notebook             | LLM level | Description |
| --------------------------- | --------- | ----------- |
| _Part_a.py                  |           |             |
| _Part_b.py                  |           |             |
| _Part_c.py                  |           |             |
| _Part_d.py                  |           |             |
| _Part_e.py                  |           |             |
| _Part_f.py                  |           |             |
| _Part_g.py                  |           |             |
| _Part_h.py                  |           |             |
| _Part_i.py                  |           |             |
| general_functions.py        |           |             |
| gradient_descent_methods.py |           |             |
| lasso_methods.py            |           |             |
| optimizer_methods.py        |           |             |
| plots.py                    |           |             |
| regression_methods.py       |           |             |
| resampling_methods.py       |           |             |
| sgd_methods.py              |           |             |
|                             |           |             |
|                             |           |             |
