#----------------------------------------------------------------------
"""
code author: Damianos Christophides 
email: chdamianos@gmail.com
This method has been presented at the 36th Congress of the European Society 
for Radiotherapy and Oncology
Citation: Christophides D, Gilbert A, Appelt AL, Fenwick J, Lilley J, 
Sebag-Montefiore D. OC-0255: Practical use of principal component analysis in 
radiotherapy planning. Radiother Oncol 2017;123:S129–30.
doi:10.1016/S0167-8140(17)30698-9. 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/
"""
#----------------------------------------------------------------------
# Import necessary modules
import PCA_bootstrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#----------------------------------------------------------------------
# Import and plot data
# Data downloaded from http://openmv.net/file/tablet-spectra.csv
data = pd.read_csv('tablet-spectra.csv', index_col=0, header=None)
data.T.plot(use_index=True, legend=False)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Absorbance')
plt.title('Spectral Data')

#----------------------------------------------------------------------
# Perform PCA
# Return principal components (pcs), eigenvectors (evs), explained_variance
# and mean_vector used to centre the data
pcs, evs, explained_variance, mean_vector = PCA_bootstrap.PCA(
    np.array(data), 0.95)

#----------------------------------------------------------------------
# Perform bootstrap analysis to extract the mean eigenvector
# and the 95% confidence interval of the eigenvector distribution
x_axis = range(data.shape[1])
x_label = 'Spectral Channel [AU]'
n_boot_samples = 100
confidence_interval = 0.95
plot_columns = 2
# define sample sizes to run the bootstrap over
sample_size = np.array([50, 100, 300, 450])
for sz in sample_size:
    data_temp = data.copy()
    data_temp = data_temp.sample(n=sz)
    fig_title = 'Number of observations: {}'.format(sz)
    boot_output = PCA_bootstrap.PCA_bootstrap(data_temp, evs, x_axis, x_label,
                                              fig_title, n_boot_samples,
                                              confidence_interval, plot_columns)

#----------------------------------------------------------------------
# Out of interest, plot first 2 prinipal components from spectral sample
plt.figure()
ax = plt.subplot(111)
ax.scatter(pcs[:, 0], pcs[:, 1])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('First two principal components')
# There seems to be 2 clusters of data

#----------------------------------------------------------------------
# Apply K-means clustering and plot results
est_clusters = KMeans(n_clusters=2)
est_clusters.fit(pcs[:, :2])
labels = est_clusters.labels_

plt.figure()
ax = plt.subplot(111)
ax.scatter(pcs[:, 0], pcs[:, 1], c=labels.astype(
    np.float), cmap=plt.get_cmap('viridis'))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('K-Means clusters of first two principal components')
