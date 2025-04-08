###############################################
#					      #
#   Sample Code for Bond Risk Premia with ML  #
#					      #
###############################################

### Summary ###
This folder contains code for running a recursive forecasting exercise that predicts bond excess returns using yield curve information and macro variables. The forecasting exercise is recursive. 

### Folder Contents ###
    -ModelComparison_Rolling.py
        Main file for running the recursive forecasting exercise
    -NNFuncBib.py
        Contains the template function for elastic net called from "ModelComparison_Rolling.py"



### Guidance ###
Due to cross-validation of hyperparameters a parallel computing architecture is strongly recommended.



### Python Package Versions ###
The following package versions have been used during development:

    - pandas == 0.25.3
    - scipy == 1.3.1
    - numpy == 1.17.2
    - statsmodels == 0.10.1
    - scikit-learn == 0.21.3 (also known as sklearn)
    - tensorflorw == 1.8.0
    - keras = 2.2.0
