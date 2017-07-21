# Summary
This repository holds code for a method of using bootstrap to test the robustness of PCA  
Version 1.0  
# Setup
To run the code you will need Python 2.7.13 (64bit) with the following modules installed:  
numpy 1.11.3  
pandas 0.19.2  
matplotlib 2.0.0  
statsmodels 0.6.1   
tqdm 4.11.2  
scikit-learn 0.18.1  
# Data preparation
1. Prepare your data in a pandas dataframe where the rows are observations and columns represent the variables you would like to perform the analysis on.  
2. Perform PCA to extract the eigenvectors. 
   You could use the provided function as:  
```
#!python
  pcs, evs, explained_variance,mean_vector=PCA_bootstrap.PCA(np.array(data),0.98)
```  
  where 'data' is the dataframe, 
  0.98 is the explained variance ratio threshold used to keep only an n-number of eigenvectors  
  'pcs' are the principal components for each observation  
  'evs' are the eigenvectors  
  'explained_variance' is the vector with the cumulative values of explained variance for each principal component  
  'mean_vector' is the mean vector used to centralise the data  
  np is the numpy library imported as 'import numpy as np'  
# How to run
See example.py  
# Output
The code provides a visualisation of eigenvectors along with the mean and 95% (or otherwise defined) confidence intervals calculated from the bootstrap process.  

The code can be run as:  
```
#!python
boot_output=PCA_bootstrap.PCA_bootstrap(data_temp,evs,n_boot_samples,confidence_interval,x_axis,plot_columns,x_label,fig_title)
```
'data_temp' is the data in a pandas dataframe where the rows are observations and columns represent the variables you would like to perform the analysis on  
'evs' are the eigenvectors extracted using PCA
'n_boot_sample' is the number of bootstrap samples to run  
'confidence_interval' is the confidence interval i.e. 0.95 corresponds to a 95% confidence interval  
'x_axis' is the vector of values to be plotted on the x-axis  
'plot_columns' is the number of columns to use in the subplots  
'x_label' is the label to be used for the x-axis of the plots  
'fig_title' is the title of the plot  
'boot_output' is a multidimensional array holding the data of the eigenvectors derived during the bootstrap process  

Below you can see an example of the graphical output  
![example n=100.png](https://bitbucket.org/repo/5qdo54A/images/2012215361-example%20n=100.png)  
![example n=300.png](https://bitbucket.org/repo/5qdo54A/images/3031254283-example%20n=300.png)  
![example n=450.png](https://bitbucket.org/repo/5qdo54A/images/1293509323-example%20n=450.png)  

As you can see as the number of observations increases the mean eigenvectors from the bootstrap have better agreement with the original eigenvectors  
Also the 95% confidence intervals get smaller  
For more information go through example.py  

# Further Information
Contact Damianos Christophides at chdamianos@gmail.com