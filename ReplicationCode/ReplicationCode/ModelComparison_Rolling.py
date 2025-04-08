#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code runs an expanding window out-of-sample forecasting exercise.

The functions used to perform the forcasts are loaded from the file NNFuncBib.py

Where data input is required a placeholder will appear that is is to be replaced by the user.
"""

# Imports
import numpy as np
import pandas as pd
import scipy.io as sio
import multiprocessing as mp
import NNFuncBib as NFB
import os
import time
import statsmodels.api as sm
from scipy.stats import t as tstat

import shutil

# User Functions


def multProcessOwnExog(NNfunc, ncpus, nMC, X, Xexog, Y, **kwargs):
    try:
        pool = mp.Pool(processes=ncpus)
        output = [pool.apply_async(NNfunc, args=(X, Xexog, Y, no,),
                                   kwds=kwargs)
                  for no in range(nMC)]
        outputCons = [p.get(timeout=3000) for p in output]
        pool.close()
        pool.join()
        time.sleep(1)

    except Exception as e:
        print(e)
        print("Timed out, shutting pool down")
        pool.close()
        pool.terminate()
        time.sleep(1)

    return outputCons


def R2OOS(y_true, y_forecast):
    import numpy as np

    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(), (np.arange(y_true.size)+1))

    # lag by one period
    y_condmean = np.insert(y_condmean, 0, np.nan)
    y_condmean = y_condmean[:-1]
    y_condmean[np.isnan(y_forecast)] = np.nan

    # Sum of Squared Resids
    SSres = np.nansum(np.square(y_true-y_forecast))
    SStot = np.nansum(np.square(y_true-y_condmean))

    return 1-SSres/SStot


def RSZ_Signif(y_true, y_forecast):

    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(), (np.arange(y_true.size)+1))

    # lag by one period
    y_condmean = np.insert(y_condmean, 0, np.nan)
    y_condmean = y_condmean[:-1]
    y_condmean[np.isnan(y_forecast)] = np.nan

    # Compute f-measure
    f = np.square(y_true-y_condmean)-np.square(y_true-y_forecast)  \
        + np.square(y_condmean-y_forecast)

    # Regress f on a constant
    x = np.ones(np.shape(f))
    model = sm.OLS(f, x, missing='drop', hasconst=True)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 12})

    return 1-tstat.cdf(results.tvalues[0], results.nobs-1)


if __name__ == "__main__":

    # =========================================================================
    #                           Settings
    # =========================================================================

    # Set True for testing, False otherwise. Reduces computational complexity.
    # TestFlag is for development purposes only.
    TestFlag = False
    if TestFlag:
        nMC = 4  # Number of networks to train in parallel
        nAvg = 2  # Number of best networks to take for model averaging
    else:
        nMC = 100
        nAvg = 10

    OOS_Start = '1989-01-31'  # Date of RHS variable where to start OOS
    data_path = 'yourpath'  # Replace with path to data as required

    HyperFreq = 4*12  # Frequency of hyper-parameter search
    NN_Params = {'Dropout': [0.1, 0.3, 0.5], 'l1l2': [0.01, 0.001]}

    # Computational Ressources: Determine Number of available cores
    ncpus = mp.cpu_count()
    print("CPU count is: "+str(ncpus))

    # Location for temporary model storage. Deleted upon completion.
    dumploc_base = './trainingDumps_'

    i = 0
    path_established = False
    while not path_established:
        dumploc = dumploc_base+str(i)
        try:
            os.mkdir(dumploc)
            print("Directory ", dumploc, " Created ")
            path_established = True
        except FileExistsError:
            print("Directory ", dumploc, " Already exists")
            i += 1

    # Set of models for training
    models = [NFB.ElasticNet_Exog_Plain, NFB.NN1LayerEnsemExog,
              NFB.NN3LayerExog]
    # Set of names to use for models. Same order as in list "models".
    modelnames = ['ElasticNetExog_Plain', 'NN1LayerEnsemExog',
                  'NN3LayerExog']

    # =========================================================================
    #                          Data Loading
    # =========================================================================

    # Data Inputs: Replace with your data as needed. X / Y variables need to be pre-aligned for forecasting.

    # RHS variable: Forward Rates, TxN, N is number of forward rates
    Xexog = Xexog_placeholder
    # RHS variable: Macro variables, TxK, K is number of macro variables
    X = X_placeholder
    # LHS Variable: Excess Returns, TxM, M is number of left handside variables forecasted simultaneously
    Y = Y_placeholder
    # Contains group number of macro variables, Kx1
    A = A_placeholder

# =========================================================================
#                   Estimation
# =========================================================================



    # Determine IS vs OS indices
    T = X.shape[0]
    tstart = np.argmax(fwdrate.index == OOS_Start)
    OoS_indeces = range(tstart, int(T))
    # Number of outputs
    M = Y.shape[1]

    if TestFlag:
        OoS_indeces = OoS_indeces[:2]

    VarSave = {}  # Storage dictionary for all results

    # Loop over models
    for modelnum, modelfunc in enumerate(models):

        Y_forecast = np.full([T, nMC, M], np.nan)
        Y_forecast_agg = np.full([T, M], np.nan)
        val_loss = np.full([T, nMC], np.nan)
        print(modelnames[modelnum])

        # Model Case Distinction
        if modelnames[modelnum] == 'NN3LayerExog':

            j = 1
            for i in OoS_indeces:

                # Determine whether to perform fresh hyper-parameter search
                if (j == 1) or (j % HyperFreq == 0):
                    refit = True
                else:
                    refit = False
                j += 1

                # Run model
                start = time.time()
                output = multProcessOwnExog(modelfunc, ncpus, nMC, X[:i+1, :],
                                            Xexog[:i+1, :], Y[:i+1, :],
                                            dumploc=dumploc, params=NN_Params,
                                            refit=refit)
                # Handle output
                val_loss[i, :] = np.array([output[k][1] for k in range(nMC)])
                Y_forecast[i, :, :] = np.concatenate([output[k][0] for k
                                                      in range(nMC)], axis=0)
                tempsort = np.argsort(val_loss[i, :])
                ypredmean = np.mean(Y_forecast[i, tempsort[:nAvg], :], axis=0)
                Y_forecast_agg[i, :] = ypredmean
                print("Obs No.: ", i, " - Step Time: ", time.time() - start)

        elif modelnames[modelnum] == 'NN1LayerEnsemExog':

            j = 1
            for i in OoS_indeces:

                if (j == 1) or (j % HyperFreq == 0):
                    refit = True
                else:
                    refit = False
                j += 1

                # Run model
                start = time.time()
                output = multProcessOwnExog(modelfunc, ncpus, nMC, X[:i+1, :],
                                            Xexog[:i+1, :], Y[:i+1, :],
                                            A=A, params=NN_Params, refit=refit,
                                            dumploc=dumploc)

                # Handle output
                val_loss[i, :] = np.array([output[k][1] for k in range(nMC)])
                Y_forecast[i, :, :] = np.concatenate([output[k][0]
                                                      for k in range(nMC)],
                                                     axis=0)
                tempsort = np.argsort(val_loss[i, :])
                ypredmean = np.mean(Y_forecast[i, tempsort[:nAvg], :], axis=0)
                Y_forecast_agg[i, :] = ypredmean
                print("Obs No.: ", i, " - Step Time: ", time.time() - start)

        elif modelnames[modelnum] == 'ElasticNetExog_Plain':
            for i in OoS_indeces:
                start = time.time()
                ypredmean = modelfunc(X[:i+1, :], Xexog[:i+1, :], Y[:i+1, :])
                Y_forecast[i, 0, :] = ypredmean
                Y_forecast_agg[i, :] = ypredmean
                print("Obs No.: ", i, " - Step Time: ", time.time() - start)

        else:
            raise Exception("Model does not match any known case.")

        # Validation Loss
        VarSave["ValLoss_"+modelnames[modelnum]] = val_loss

        # Fitted / Forecasted Values (Averaged)
        VarSave["Y_forecast_agg_"+modelnames[modelnum]] = Y_forecast_agg

        # Fitted / Forecasted Values (From models trained in parallel)
        VarSave["Y_forecast_"+modelnames[modelnum]] = Y_forecast

        # Mean Squared Error
        VarSave["MSE_"+modelnames[modelnum]] = np.nanmean(np.square(Y-Y_forecast_agg), axis=0)

        # R2-00S
        VarSave["R2OOS_"+modelnames[modelnum]] = np.array(
                [R2OOS(Y[:, k], Y_forecast_agg[:, k]) for k in range(np.size(Y, axis=1))])

        print('R2OOS: ', VarSave["R2OOS_"+modelnames[modelnum]])

        # Rapach, Strauss, Zhou (2010) - Significance R^2
        VarSave["R2OOS_pval_"+modelnames[modelnum]] = \
            np.array([RSZ_Signif(Y[:, k], Y_forecast_agg[:, k])
                      for k in range(np.size(Y, axis=1))])

        savesuccess_flag = False
        while not savesuccess_flag:
            # Save new result to SOA file
            try:
                VarSaveSOA = sio.loadmat('ModelComparison_Rolling_SOA.mat')
                VarSaveSOA.update(VarSave)
                sio.savemat('ModelComparison_Rolling_SOA.mat', VarSaveSOA)
                savesuccess_flag = True
                print('Updated SOA file')
            except FileNotFoundError:
                sio.savemat('ModelComparison_Rolling_SOA.mat', VarSave)
                savesuccess_flag = True
                print('Created new SOA file')

    # Delete Dumploc
    try:
        shutil.rmtree(dumploc)
        print('Removed dir: '+dumploc+' succesfully')
    except FileNotFoundError:
        print('Directory: '+dumploc+' could not be removed')
