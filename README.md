## Running code

main.py

## datasets
(1)**Simulation Dataset**  
sim1.txt: 5 nodes  
sim2.txt: 10 nodes  
sim3.txt: 15 nodes  
sim4.txt: 50 nodes  

(2)**Synthetic Dataset**  
sim200.csv: 200 nodes  
sim500.csv: 500 nodes    

(3)**Real World fMRI Dataset**  
ALLASD1_cc200-nor.txt: normal   
ALLASD1_cc200-pat.txt: patient  

## Install Requirements

The repo use python code. To install the python dependencies, run:

```
conda env create -f environment.yml
```
The most important python dependencies are `numpy`, `torch` and `functorch` which can be easily installed manually 
if one prefer not to use conda.

