# README

# Manuscript title: Local signal detection on irregular domains with generalized varying coefficient models

## Abstract

In spatial analysis, it is essential to understand and quantify spatial or temporal hetero-
geneity. This paper focuses on the generalized spatially varying coefficient model (GSVCM),
a powerful framework to accommodate spatial heterogeneity by allowing regression coeffi-
cients to vary in a given spatial domain. We propose a penalized bivariate spline method
for detecting local signals in GSVCM. The key idea is to use bivariate splines defined on
triangulation to approximate nonparametric varying coefficient functions and impose a local
penalty on L2 norms of spline coefficients for each triangle to identify null regions of zero
effects. Moreover, we develop model confidence regions as the inference tool to quantify the
uncertainty of the estimated null regions. Our method partitions the region of interest using
triangulation and efficiently approximates irregular domains. In addition, we propose an ef-
ficient algorithm to obtain the proposed estimator using the local quadratic approximation.
We also establish the consistency of estimated nonparametric coefficient functions and the
estimated null regions. The numerical performance of the proposed method is evaluated in
both simulation cases and real data analysis.

## Reproducibility

In this Markdown file, we provide some information related to reproducibility, which mainly covers functionality of each Matlab code script and what plots they would generate.

# Requirement

Our Matlab version and computation environment are listed as below

**Matlab Version**: R2018a

**CPU**: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz 2.59 GHz

**Ram**: 16 GB

# File Overview

All Matlab code files and plots are stored at two folders: *Matlab Code* and *Visualization*. They contain matlab code scripts and associated plots respectively. Under the folder *Matlab Code*, we have

- The file `github_numerical_simulation` is for simulation code for the case when the response variable is continuous.

- The file `github_Poisson_simulation2.m` is for the case when the response variable follows a Poisson distribution.

- And `github_BJ.m` and `github_BJ_plots.m` are for the case when we use Beijing housing price as the application of our proposed method. Specifically, the relation between Matlab code and plots are listed as below

The script `github_numerical_simulation` generates: `TRI-BIC.jpg`, `TRI-DEN.jpg`, `TRI-SPARSE.jpg`, `TRI-UNIF.jpg`, `Fig - Estimated Zero Regions of Continuous Simulation`, `Fig - MCR of Continuous Simulation`.

# Running Time

## Numerical Simulation

For continuous response simulation

## Beijing Housing Price Application

For practical application

# Dataset Access

We use a Beijing housing price dataset for practical application. The raw dataset source is: https://www.kaggle.com/datasets/ruiqurm/lianjia

However, we made some changes on the raw dataset, which mainly is about adding interaction of variables and implement standardization. Because github has a limit of file size, so we upload dataset files to a public Kaggle dataset address: https://www.kaggle.com/datasets/chengzhuzhang/transformed-beijing-housing-price-from-lianjia

There are two CSV files in the before-mentioned Kaggle dataset address, which need to be downloaded to current working directory as sourcing datasets. Specifically, they are

* `new BJ house.csv`: this CSV file is the raw source dataset I got from Kaggle dataset. This dataset contains 293,963 data points and 28 columns. In the Matlab code, only column 3 (long) and column 4 (lat) are necessary. The location information is used to determine whether the house falls into the manual crafted triangulation. Only valid data points are kept for down-streaming fitting. The detailed list of column names is contained in the second line of the Matlab script `github_BJ.m`.

* `new BJ stepAIC design matrix std.csv`: this CSV file is transformed from `new BJ house.csv`, which contains all variables including raw variables and interaction of raw variables. They are picked via AIC and stepwise selection method from the entire set consists of all main effects and interactions of all covariates. This step is implemented in R studio (but not added to the current folder). You can check the first row of this CSV file to know what it represents. The first column is called “intercept” and would always be 1. This dataset is already standardized.

## Continuous Response Simulation

The main code script is `github_numerical_simulation.m` which can be found in the subfolder "Matlab Code" and supplementary TRI plots are stored at the subfolder "Visualization".

## Poisson Response Simulation

The main code script is `github_Poisson_simulation2.m` which can be found in the subfolder "Matlab Code".

## Beijing Housing Price Application

The main code script is `github_BJ.m` which contains fitting for cross-validation part. The scatterplot and summary table of 5-fold cross-validation are generated in this script. Furthermore, we included another file `github_BJ_explore.m` which contains the early stage exploration we made about the Beijing housing price dataset. It includes exploration of the raw dataset and the fitting results from standardized dataset and district-specific linear regression.

# Helper function list

Besides main Matlab scripts that are mentioned above, you also need other helper functions which serves some basis functions related to Triangulation. For example, we directly use the function provided by Larry L. Schumaker to generate triangulation. We list them and associated resources as below

- basic helper functions from Larry L. Schumaker: `basis.p`, `choose.p`, `getindex.p`, `trilists.p`, `dcircle.m`, `ddiff.m`, `distmesh2d.m`, `drectangle.m`, `fixmesh.m`

- `CZ_SPL_est.m`: created by Chengzhu Zhang, which takes a list of location (X, Y) and Triangulation information as input. The output contains two parts. The first output is values of associated Bernstein basis polynomials of degree d, while the second output is the list of order numbers indicating valid points that fall into the Triangulation. More details can be found in the comments contained in the file.

- `update_p_b_hat_2.m`: created by Chengzhu Zhang, which takes unpenalized estimator "b_hat", design matrix "mat_Z", response "Z" and Triangulation related values as input, to calculate SCAD penalized estimator "p_b_hat". This function is used for continuous response variable.

- `update_p_b_hat_poisson.m`: created by Chengzhu Zhang, which takes unpenalized estimator "b_hat", design matrix "mat_Z", response "Z" and Triangulation related values as input, to calculate SCAD penalized estimator "p_b_hat". This function is used for discrete response variable that follows a Poisson distribution.
