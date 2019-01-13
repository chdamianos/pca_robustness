"""
code author: Damianos Christophides 
email: chdamianos@gmail.com
This method has been presented at the 36th Congress of the European Society 
for Radiotherapy and Oncology
Citation: Christophides D, Gilbert A, Appelt AL, Fenwick J, Lilley J, 
Sebag-Montefiore D. OC-0255: Practical use of principal component analysis in 
radiotherapy planning. Radiother Oncol 2017;123:S129:30.
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


def PCA(arr, thre):
    """
    Function to implement PCA
    Input: array with observations as rows and variables as columns, 
    threshold to use for explained variance, e.g. 0.95 is 95%
    Output: Principal components (PC), Eigenvectors, Cumulative explained variance,
    mean vector used to centre data
    """

    import numpy as np
    # Centre the data
    mean_arr = np.mean(arr, axis=0)
    arr_central = arr - mean_arr
    # Calculate covariance matrix
    m, n = np.shape(arr_central)
    cov_matrix = (1.0 / (m - 1.0)) * np.dot(arr_central.T, arr_central)
    # Perform singular value decomposition to derive eigenvectors and eigenvalues
    u, s, v = np.linalg.svd(cov_matrix)
    assert np.allclose(cov_matrix, np.dot(u * s, v))
    # Tranform data to derive principal compoenents
    # it doesn't matter if you use u or v since one is the inverse rotation of the other
    # it's like saying that rotation of +35o is different from -35o
    # see diagram in https://en.wikipedia.org/wiki/Singular_value_decomposition
    EigenVectors = v
    PC = np.dot(arr_central, EigenVectors)
    # Calculate cumulative explained variance from eigenvalues
    explained_variance = s / np.sum(s)
    explained_variance = np.cumsum(explained_variance)
    thre_idx = np.where(explained_variance >= thre)[0][0]

    return PC[:, :thre_idx + 1], EigenVectors[:, :thre_idx + 1], explained_variance, mean_arr


def conf_interval(arr, alpha):
    """
    Function to produce confidence intervals for eigenvectors
    Input: array with rows as eigenvectors, alpha to use in confidence interval 
    e.g. 0.05 corresponds to 95% confidence
    Output: mean eigenvector, lower % confidence limit, upper % confidence limit
    """
    #----------------------------------------------------------------------
    # Import necessary modules
    import numpy as np
    import statsmodels.api as sm

    no_rows, n_cols = arr.shape
    lower_val = np.zeros(n_cols)
    upper_val = np.zeros(n_cols)
    mean_val = np.zeros(n_cols)

    #----------------------------------------------------------------------
    # Calculation of confidence intervals from distribution of eigenvector values
    # and calculation of mean
    count_ = 0
    for column in arr.T:
        # Calculate empirical cumulative distribution function for each variable
        # i.e column in the dataset
        ecdf = sm.distributions.ECDF(column)
        x = np.linspace(np.min(column), np.max(column), num=len(column))
        y = ecdf(x)
        # From the ecdf calculated confidence intervals based on alpha value
        # using interpolation
        lower_val[count_] = np.interp(alpha / 2., y, x)
        upper_val[count_] = np.interp(1.0 - (alpha / 2.), y, x)
        # Calculate mean value
        mean_val[count_] = np.mean(column)
        count_ = count_ + 1

    return mean_val, lower_val, upper_val


def PCA_bootstrap(data, evs, x, x_lbl, tlt, n_boots=100, ci_=0.95, plot_cols=2):
    """
    Function to peform eigenvector bottstrap confidence analysis
    Input: data: dataframe with observations as rows, evs: original eigenvectors,
    n_boots: number of bootstrap samples, ci: confidence interval 
    (0.95 corresponds to 95%), x: vectors with x values, x_lbl: Labels of x-axis plot,
    tlt: plot title
    Output: plots original eigenvectors with mean and confidence intervals
    """
    #----------------------------------------------------------------------
    # Import necessary modules
    import numpy as np
    from sklearn.utils import resample
    import matplotlib.pyplot as plt
    import tqdm

    alpha_ = 1.0 - ci_  # set alpha level based on confidence interval
    rows, cols = evs.shape
    test_arr = np.arange(0, len(data))

    #----------------------------------------------------------------------
    # Empty array to hold the boostrap-calculated eigenvectors
    empty_boot = np.zeros([cols, rows, n_boots])

    #----------------------------------------------------------------------
    # Bootstrap process
    for i in tqdm.tqdm(range(n_boots), desc='PCA Bootstrap with sample size {}'.format(len(data))):
        # Copy and take bootstrap sample of data
        temp_ = np.copy(data)
        test_sample = resample(test_arr, n_samples=len(test_arr), replace=True)
        arr_sample = temp_[test_sample]
        # Perform PCA
        PCs_, EV_, exp_var_, mean_arr_ = PCA(np.array(arr_sample), 0.99999)
        for j in range(cols):
            # If necessary invert eigenvectors so they are pointing in the same
            # direction based on their maximum magnitude
            sign_evs_idx = np.argmax(np.abs(evs[:, j]))
            # Note that using the maximum amplitude may not be suitable for all cases
            # instead use the index of a peak or trough in the original eigenvectors
            # relevant to your analysis
            if np.sign(evs[sign_evs_idx, j]) == np.sign(EV_[sign_evs_idx, j]):
                empty_boot[j, :, i] = EV_[:, j]
            else:
                empty_boot[j, :, i] = -1.0 * EV_[:, j]

    #----------------------------------------------------------------------
    # Plots
    plot_rows = int(np.ceil(cols / float(plot_cols)))

    fig, ax_ = plt.subplots(plot_rows, plot_cols, sharex=True, sharey=False,
                            figsize=(8, 6))
    axs = ax_.ravel()
    for k in range(cols):
        # Extract mean, upper and lower confidence intervals from
        # bootstrap-calculated eigenvectors
        mean_, low_, up_ = conf_interval(empty_boot[k, :, :].T, alpha_)
        # Plot original eigenvector, mean and confidence intervals
        axs[k].plot(x, evs[:, k], 'k', label='Original Eigenvector')
        axs[k].plot(x, mean_, '--k', label='Mean Eigenvector')
        axs[k].fill_between(x, low_, up_, alpha=0.5,
                            edgecolor='#000000', facecolor='#000000')
        axs[k].set_title('Eigenvector ' + str(k + 1))

        if k % plot_cols == 0:
            axs[k].set_ylabel('Weight', fontsize=12, fontweight='bold')
        if k >= ((plot_rows - 1) * plot_cols):
            axs[k].set_xlabel(x_lbl, fontsize=12, fontweight='bold')

    axs[0].legend()
    fig.suptitle(tlt, fontsize="x-large")

    #----------------------------------------------------------------------
    # Return bootstrap-calculated eigenvectors
    return empty_boot
