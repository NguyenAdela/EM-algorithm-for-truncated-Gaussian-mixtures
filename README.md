# EM-algorithm-for-truncated-Gaussian-mixtures

This repository contains the implementation of the expectation-maximization (EM) algorithm for truncated Gaussian mixtures in Python. It is a part of the master thesis **The EM algorithm of truncated Gaussian mixtures** which is also available here: [thesis.pdf](Thesis.pdf).

The comparation of implemented algorithm with the algorithm used in the article **EM algorithms for multivariate Gaussian mixture models with truncated and censored data.** *Lee, G., & Scott, C. (2012). Computational Statistics & Data Analysis, 56(9), 2816-2829.* is included.

## Requirements
The script is written in Python 3.10.4. R with library truncnorm is also used. Please put the path of installed R with the mentioned library in [.env](.env) file as R_PATH.

Library requirements can be found in [requirements.txt](requirements.txt).

Change to the directory where [requirements.txt](requirements.txt) is located and activate your virtualenv. Then run: `pip install -r requirements.txt` in your shell.