# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* This code repository demonstrates how bootstrap could be used to test the robustness of PCA
* Version 1.0

### How do I get set up? ###
* Prepare your data in a pandas dataframe where the rows are observations and columns represent the variables you would like to perform the analysis on.

First you need to perform PCA to extract the eigenvectors. 
You could use the provided function as:

```
#!python
pcs, evs, explained_variance,mean_vector=PCA_bootstrap.PCA(np.array(data),0.98)
```

where 'data' is the dataframe 
0.98 is the explained variance ratio threshold to keep only n-number of eigenvectors

'pcs' are the principal components for each observation
'evs' are the eigenvectors
'explained_variance' is the vector with the cumulative values of explained variance for each principal component
'mean_vector' is the mean vector used to centralise the data
from there variables we only need 'evs'

np is the numpy library imported as 'import numpy as np'

* Dependencies
You will need the following Python libraries
numpy, pandas, matplotlib, statsmodels, tqdm, sklearn
* How to run tests
See example.py 
* Deployment instructions
Download the source code and use as appropriate for your application

### Who do I talk to? ###
* Damianos Christophides
* chdamianos@gmail.com