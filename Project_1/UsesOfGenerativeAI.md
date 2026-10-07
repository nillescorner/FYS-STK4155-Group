This .md file provides a full overview of our usage of LLMs. Claude Sonnet 5.5 Medium was used in formatting this file.

Guide for labeling as provided in MachineLearningUiO:
| Level | Label | Meaning |
| --- | --- | --- |
| 0 | None | No LLM assistance |
| 1 | Debugging | LLM helped identify a bug; fix implemented by you |
| 2 | Snippet | LLM provided a function, loop, or short block; you integrated and tested it |
| 3 | Skeleton | LLM generated the overall structure of a script/class; you filled in domain-specific logic |
| 4 | Substantial | LLM wrote the majority of a file; you adapted, tested, and commented it |


**Models used.**

- Model: Sonnet 5, Effort: Medium until September 27 as Sonnet 5.5 launched September 28.
- Model: Sonnet 5.5, Effort: Medium September 28-31.
- Model: Opus 5.5 Effort: Medium  October 1st - 5th.
- Model: GPT.6 Luna Effort: Medium September 28 - October 7th 
- Model: MAI-Code-1.1-Flash Effort: Low September 28 - October 7th 

**Text writing and editing.**

| Section | Level | Notes |
| --- | --- | --- |
| Abstract | 1 | LLM helped with restructuring sentences to shorten down the abstract. |
| Introduction | 1 | LLM helped with spellchecking and restructuring sentences from ideas we supplied. |
| Methods | 2 | LLM rewrote and restructured and sentences from ideas we supplied. LLM was used as an academic sparring partner to better understand unclear concepts from the material provided |
| Results | 1 | Output data from code was sent into LLM to convert to Table format. Figures were sent into LLM to convert into latex syntax. |
| Discussion | 2 | LLM was used as an academic sparring partner when interpreting results to see if they were correct with the theory from methods. LLM was also used to help with restructuring sentences that were hard to read. |
| Conclusion | 1 | Language corrections |


**Code generation and assistance.**

| File / notebook | LLM level | Description |
| --- | --- | --- |
| `__init__.py` | 1 | How to access modules from other files |
| `_Part_a.py` | 1 | LLM assisted with how to plot the heat map and how to properly extract the lowest MSE and highest R2 score from the heatmap. LLM also checked if independently written code worked as expected. No changes were made. |
| `_Part_b.py` | 1 | LLM assisted with how to plot the heat map and how to properly extract the lowest MSE and highest R2 score from the heatmap.  LLM also checked if independently written code worked as expected. No changes were made. |
| `_Part_c.py` | 1 | LLM checked if independently written code worked as expected. No changes were made. |
| `_Part_d.py` | 2 | LLM assisted in plotting the error bars for cross validation and how to print the output for bootstrap and CV lowest MSE. LLM assisted with for loop to print the values that make up Table \ref{tab:part_d_ridge_cv_lambda}. LLM assisted with for loop to plot Figure \ref{fig:part_d_ridge_vs_lambdas}. LLM also checked if independently written code worked as expected. LLM notified that maxdegree had changed in our code compared to cross\_validation default, this was fixed by us. |
| `_Part_e.py` | 1 | LLM helped with debugging code, fixed minor error that didn't let code run. |
| `_Part_f.py` | 1 | LLM used for minor debugging |
| `_Part_g.py` | 2 | LLM helped with the comparison of different Lasso methods. |
| `_Part_h.py` | 2 | LLM looked over code, and flagged that sgd for the adaptive optimisers hadn't been done yet, and showed a suggestion to how this could be implemented. |
| `_Part_i.py` | 2 | Writing plotting plot\_best\_model\_fits |
| `colors.py` | 1 | Colors throughout the program were manually written down by hex codes, LLM provided the name of the color for these hex codes. |
| `general_functions.py` | 1 | All functions in file were written by us. LLM was used to check if the code worked as expected. No changes were made. |
| `gradient_descent_methods.py` | 1 | LLM helped debug and comment code. |
| `lasso_methods.py` | 1 | LLM helped debug and comment code. |
| `main_test.py` | 2 | How to access modules from other files. Cleaning up visualisation of how n resamples using kfold resampling affects the bias variance decomposition |
| `optimizer_methods.py` | 1 | LLM helped debug and comment code. |
| `plots.py` | 4 | LLM wrote all the functions in this file after being provided with information of what we wanted to plot and how. Functions were commented manually but with basis in the few comments provided by the LLM. |
| `README.md` | 2 | Asked LLM to make a structure of the program folders and help with syntax when writing the README. |
| `regression_methods.py` | 1 | All functions in file were written by us. LLM was used to check if the code worked as expected. No changes were made |
| `requirements.txt` | 4 | Asked LLM to create a text file that contained the requirements needed to run the code in Project 1 |
| `resampling_methods.py` | 3 | K-fold resampling algorithm mostly LLM made Implemented test\_wholedataset function in bootstrap\_resampling based on selfmade version in kfold\_resampling. Ran through for glaring bugs Assisted with doc strings for train\_through\_gridsearchCV, mse\_decomposer and grid\_search impementation of std calculations in other methods than cross\_validation, to mirror usage already defined there |
| `sgd_methods.py` | 1 | LLM helped debug and comment code. |
