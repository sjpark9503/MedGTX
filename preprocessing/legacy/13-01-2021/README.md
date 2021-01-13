# MIMIC-KG & MIMIC-Subgraphs
This code convert MIMIC-III tables into Knowledge Graph triples & subgraphs that contains a single admission node.

## Datasets
### 0. Prepare MIMIC-III
First, you need to access the MIMIC-III data. This requires certification from https://mimic.physionet.org/. 
Then, you must place the MIMIC-III csv files under mimic_tables

### 1. Build KG 
Run table2triple.ipynb

### 2. Extract subgraphs from KG
Run triple2subgraph.ipynb
This will return results of depth-first-search(DFS) over KG starting from an admission node where subgraph_withrel contains relation and subgraph_norel is just a sequence of nodes.

<br/>

---

<br/>

# MIMIC-NOTEEVENTS
This code extract the specific section from discharge summary in MIMIC-III noteevents.  
For now, it only provides the p_section which means prescriptions.  

## Datasets
### 0. Prepare `NOTEEVENTS.csv` in MIMIC-III

### 1. Extract the specific section from discharge summary in `NOTEEVNETS.csv`
