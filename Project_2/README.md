## Project 1 - FYS-STK4155

Hello and welcome to the Project 2 folder in our FYS-STK4155 repository for the group consisting of Pernille Xu Amundsen, Maria Andersen and Hannah Westgaard. 


The project is structured as:

---
```
Project_2/
├── Code/
│   ├── intermediary_results/   
│   ├── models/              
│   ├── _Part_a.py                  
│   ├── _Part_b.py                  
│   ├── _Part_c.ipynb                 
│   ├── _Part_d.ipynb                
│   ├── _Part_e.py                 
│   ├── _Part_e_coarsegrid.py
│   ├── _Part_e_layers_penalties_bestcheapest.py               
│   ├── _Part_e_layers_penalties.py
│   ├── _Part_e_TF_retraining.py
│   ├── _Part_e.ipynb
│   │
│   ├── general_functions.py
│   ├── linear_regression.py
│   ├── neural_network.py        
│   ├── optimizer.py       
│   └── plot_colors.py              # Color scheme and font sizes
├── figs/                           # Generated figures used in Report
├── report/                         # Finalized Report (PDF)
└── README.md
├── requirements.txt                # Requirements to run this program
```
---

#### Installation
Python 3.10 or newer is recommended. Our group used a mixture of 3.11.9 and 3.12.7. Install the dependencies via:

pip install -r requirements.txt 

---

#### Running the code
All scripts import functions, variables etc from the same folder, thus use

```bash
cd Code 
```
in order to import relevant functions and variables from files within the same Code folker.

Each _Part_x.py or _Part_x.ipynb file corresponds to the matching part of the project description and can be run independantly.
.ipynb have been already run and have output already available. For the .py files, run

```bash
python _Part_x.py
```

to run the program.


## Use of LLMs

See the declaration at the end of the report. Code sections written with LLM assistance of level 2 or above are marked with comments (`# LLM Assisted`) in the source files.