#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prototype functions for ML models.

All neural networks rely on NNGridSearchWrapper.
Each neural network architecture is derived from a generic function for the
kind of network (NNExogGeneric, NNEnsemExogGeneric). The functions
NN3LayerExog / NN1LayerEnsemExog call the corresponding generics and modify them
with the particular architecture (e.g. 3 layers with x-y-z nodes).

Nomenclature:
    - Xexog refers to the yields variables
    - X refers to the macro variables
    - A contains the variable groups for the macro variables

"""

import numpy as np
from sklearn.model_selection import ParameterGrid


def NNGridSearchWrapper(NNfunc, X, Y, no, params=None, refit=None,
                        dumploc=None, **kwargs):

    """
    Performs the gridsearch over the parameter dictionary params.
    """

    # Construct grid of parameters from dictionary, containing param ranges
    paramgrid = list(ParameterGrid(params))

    # Loop over all param grid combinations and save val_loss
    val_loss = list()
    for i, param_i in enumerate(paramgrid):
        _, val_loss_temp = NNfunc(X, Y, no,
                                     dropout_u=param_i['Dropout'],
                                     l1l2penal=param_i['l1l2'],
                                     refit=True, dumploc=dumploc,
                                     **kwargs)
        val_loss.append(val_loss_temp)

    # Determine best model according to grid-search val_loss
    bestm = np.argmin(val_loss)

    # Fit best model again
    Ypred, val_loss = NNfunc(X, Y, no, dropout_u=paramgrid[bestm]['Dropout'],
                                l1l2penal=paramgrid[bestm]['l1l2'],
                                refit=True, dumploc=dumploc,
                                **kwargs)

    return Ypred, val_loss


def NNExogGeneric(X, Y, no, dropout_u=None, l1l2penal=None, refit=None,
                  dumploc=None, **kwargs):

    """
    This model fits a vanilla neural network on the macro variables and
    merges it with the yields variables in the last hidden layer.
    """

    if dumploc == None:
        raise ValueError('Missing Dumploc argument')

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, BatchNormalization
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers.merge import concatenate
    from keras.optimizers import SGD
    from keras.models import load_model
    from keras import regularizers

    Xexog = kwargs['Xexog']
    archi = kwargs['archi']

    #Split Data for Test and Training
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]

    X_test = X[-1,:].reshape(1,-1)
    Xexog_test = Xexog[-1,:].reshape(1,-1)


    #Scale the predictors for training
    Xscaler_train =  MinMaxScaler(feature_range=(-1,1))
    X_scaled_train = Xscaler_train.fit_transform(X_train)

    Xexog_scaler_train =  MinMaxScaler(feature_range=(-1,1))
    Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)

    # Keras requires 3D tuples for training.
    X_scaled_train = np.expand_dims(X_scaled_train, axis=1)
    Xexog_scaled_train = np.expand_dims(Xexog_scaled_train, axis=1)

    Y_train = np.expand_dims(Y_train, axis=1)

    # seed numpy and tf
    tf.set_random_seed(no)
    np.random.seed(no)


    # Define Model Architecture
    if refit:
        # Base model for macro variables
        n = len(archi)
        layers = dict()
        for i in range(n+1):
            if i == 0:
                layers['ins_main'] = Input(shape=(1,X_scaled_train.shape[2]))
            elif i == 1:
                layers['dropout'+str(i)] = Dropout(dropout_u)(layers['ins_main'])
                layers['hidden'+str(i)] = Dense(archi[i-1],kernel_regularizer=regularizers.l1_l2(l1l2penal),bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu')(layers['dropout'+str(i)])
            elif i > 1 & i <= n:
                layers['dropout'+str(i)] = Dropout(dropout_u)(layers['hidden'+str(i-1)])
                layers['hidden'+str(i)] = Dense(archi[i-1],kernel_regularizer=regularizers.l1_l2(l1l2penal),bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu')(layers['dropout'+str(i)])

        # Model for yield variables
        layers['ins_exog'] = Input(shape=(1,Xexog_scaled_train.shape[2]))

        # Merge macro / yield networks
        layers['merge'] = concatenate([layers['hidden'+str(n)], layers['ins_exog']])
        layers['dropout_final'] = Dropout(dropout_u)(layers['merge'])
        layers['BN'] = BatchNormalization()(layers['dropout_final'])
        layers['output'] = Dense(Y_train.shape[2],bias_initializer='he_normal',
                                 kernel_initializer='he_normal')(layers['BN'])

        model = Model(inputs=[layers['ins_main'], layers['ins_exog']], outputs=layers['output'])

        # Compile model
        sgd_fine = SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss',min_delta=1e-6,
                                      patience=20,verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',
                              monitor='val_loss',save_best_only=True)

        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit([X_scaled_train, Xexog_scaled_train] , Y_train, epochs=500,
                          callbacks=[earlystopping,mcp], validation_split=0.15,
                          batch_size=32, shuffle=True, verbose=0)

        # Retrieve the best model as per early stopping
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        # Save model
        model.save(dumploc+'/BestModel_'+str(no)+'.hdf5')

    else:
        # Retrieve model architecture and retrain
        model = load_model(dumploc+'/BestModel_'+str(no)+'.hdf5')
        sgd_fine = SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=1e-6,
                                      patience=20, verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',
                              monitor='val_loss', save_best_only=True)

        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit([X_scaled_train, Xexog_scaled_train], Y_train,
                          epochs=500, callbacks=[earlystopping,mcp],
                          validation_split=0.15, batch_size=32, shuffle=True,
                          verbose=0)

        # Retrieve the best model as per early stopping
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')


    # Scale the data for testing using the in-sample transformation from earlier
    X_scaled_test = Xscaler_train.transform(X_test)
    Xexog_scaled_test = Xexog_scaler_train.transform(Xexog_test)
    X_scaled_test = np.expand_dims(X_scaled_test, axis=1)
    Xexog_scaled_test = np.expand_dims(Xexog_scaled_test, axis=1)

    # Make out-of-sample prediction on the unseen observations
    Ypred = model.predict([X_scaled_test,Xexog_scaled_test])
    Ypred = np.squeeze(Ypred,axis=1)

    return Ypred, np.min(history.history['val_loss'])


def NN3LayerExog(X, Xexog, Y, no, params=None, refit=None, dumploc=None):
    # Define number of nodes for each of the layers. Amend no of layers here.
    archi = [32, 16, 8]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNExogGeneric, X, Y, no,
                                               params=params,
                                               refit=True, dumploc=dumploc,
                                               archi=archi, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNExogGeneric(X, Y, no,
                                        refit=False, dumploc=dumploc,
                                        archi=archi, Xexog=Xexog)

    return Ypred, val_loss


def NNEnsemExogGeneric(X, Y, no,  dropout_u=None, l1l2penal=None, refit=None,
                       dumploc=None, **kwargs):

    """
    This model fits a neural network for each group of macro variables and
    ensembles those macro networks. It also adds the yields variables in the
    last hidden layer.
    """

    if dumploc == None:
        raise ValueError('Missing Dumploc argument')

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, BatchNormalization
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers.merge import concatenate
    from keras.optimizers import SGD
    from keras.models import load_model
    from keras import regularizers

    A = kwargs['A']
    archi = kwargs['archi']
    Xexog = kwargs['Xexog']

    # Split Data for Test and Training
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    X_test = X[-1,:].reshape(1,-1)
    Xexog_test = Xexog[-1,:].reshape(1,-1)


    #Scale the predictors for training
    Xscaler_train = MinMaxScaler(feature_range=(-1,1))
    X_scaled_train = Xscaler_train.fit_transform(X_train)
    X_scaled_train = np.expand_dims(X_scaled_train, axis=1)
    X_scaled_test = Xscaler_train.transform(X_test)

    Xexog_scaler_train = MinMaxScaler(feature_range=(-1,1))
    Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)
    Xexog_scaled_test = Xexog_scaler_train.transform(Xexog_test)

    Xexog_scaled_train = np.expand_dims(Xexog_scaled_train, axis=1)
    Xexog_scaled_test = np.expand_dims(Xexog_scaled_test, axis=1)

    Y_train = np.expand_dims(Y_train, axis=1)

    # Split X_train / X_test by group
    X_scaled_train_grouped = []
    X_scaled_test_grouped = []
    n_groups = len(np.unique(A))
    for i, group in enumerate(np.unique(A)):
        temp = X_scaled_train[:,A==group]
        X_scaled_train_grouped.append(np.expand_dims(temp, axis=1))
        temp = X_scaled_test[A==group].reshape(1,-1)
        X_scaled_test_grouped.append(np.expand_dims(temp, axis=1))


    # Seed numpy and tf
    tf.set_random_seed(no)
    np.random.seed(no)

    # Define Model Architecture
    if refit:
        n = len(archi)
        layers = dict()

        # Model for macro variables
        for i in range(n+1):
            if i == 0:
                layers['ins_main'] = [Input(shape=(1,X_scaled_train_grouped[j].shape[2])) for j in range(n_groups)]
            elif i == 1:
                layers['dropout'+str(i)] = [Dropout(dropout_u)(input_tensor) for input_tensor in layers['ins_main']]
                layers['hiddens'+str(i)] = [Dense(archi[i-1], activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1l2penal))(input_tensor) for input_tensor in layers['dropout'+str(i)]]
            elif i > 1 & i <= n:
                layers['dropout'+str(i)] = [Dropout(dropout_u)(input_tensor) for input_tensor in layers['hiddens'+str(i-1)]]
                layers['hiddens'+str(i)] = [Dense(archi[i-1], activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1l2penal))(input_tensor) for input_tensor in layers['dropout'+str(i)]]

        # Model for yiel variables
        layers['ins_exog'] = Input(shape=(1,Xexog_scaled_train.shape[2]))
        # Merge macro group models
        layers['merge'] = concatenate(layers['hiddens'+str(n)])
        # Merge macro and yields models
        layers['merge1'] = concatenate([layers['merge'], layers['ins_exog']])
        layers['dropout_final'] = Dropout(dropout_u)(layers['merge1'])
        layers['BN'] = BatchNormalization()(layers['dropout_final'])
        layers['output'] = Dense(Y_train.shape[2], kernel_initializer='he_normal')(layers['BN'])

        model = Model(inputs=layers['ins_main']+[layers['ins_exog']],
                      outputs=layers['output'])


        # Compile model
        sgd_fine = SGD(lr=0.015, momentum=0.9, decay=0.01, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=1e-6,
                                      patience=20, verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',
                              monitor='val_loss', save_best_only=True)

        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit(X_scaled_train_grouped+[Xexog_scaled_train], Y_train,
                          epochs=500, callbacks=[earlystopping,mcp],
                          validation_split=0.15, batch_size=32, shuffle=True,
                          verbose=0)

        # Retrieve the best model as per early stopping
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')
        # Save model
        model.save(dumploc+'/BestModel_'+str(no)+'.hdf5')

    else:
        model = load_model(dumploc+'/BestModel_'+str(no)+'.hdf5')
        sgd_fine = SGD(lr=0.015, momentum=0.9, decay=0.01, nesterov=True)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=1e-6,
                                      patience=20,verbose=0, mode='auto')
        mcp = ModelCheckpoint(dumploc+'/BestModelWeights_'+str(no)+'.hdf5',
                              monitor='val_loss', save_best_only=True)

        model.compile(loss='mean_squared_error', optimizer=sgd_fine)
        history=model.fit(X_scaled_train_grouped+[Xexog_scaled_train], Y_train,
                          epochs=500, callbacks=[earlystopping,mcp],
                          validation_split=0.15, batch_size=32, shuffle=True,
                          verbose=0)

        # Retrieve the best model as per early stopping
        model.load_weights(dumploc+'/BestModelWeights_'+str(no)+'.hdf5')

    # Perform Forecast and Test Model
    Ypred = model.predict(X_scaled_test_grouped+[Xexog_scaled_test])
    Ypred = np.squeeze(Ypred,axis=1)


    return Ypred, np.min(history.history['val_loss'])


def NN1LayerEnsemExog(X,Xexog,Y,no,params=None, refit=None, dumploc=None, A=None):
    archi = [1]

    # Perform grid-search over params
    if refit:
        Ypred, val_loss  = NNGridSearchWrapper(NNEnsemExogGeneric, X, Y, no,
                                               params=params,refit=True,
                                               dumploc=dumploc,
                                               archi=archi, A=A, Xexog=Xexog)
    # Use existing model
    else:
        Ypred, val_loss = NNEnsemExogGeneric(X, Y, no,
                                    refit=False, dumploc=dumploc,
                                    archi=archi, A=A, Xexog=Xexog)

    return Ypred, val_loss


def ElasticNet_Exog_Plain(X,Xexog,Y):
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit

    # Split data into training and test
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    X_test = X[-1,:]
    X_test = X_test.reshape(1, -1)
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)

    # Scale Inputs for Training
    Xscaler_train = StandardScaler()
    X_train = Xscaler_train.fit_transform(X_train)
    X_test = Xscaler_train.transform(X_test)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)


    # Construct Validation sample as last 15% of sample
    N_train = int(np.round(np.size(X_train,axis=0)*0.85))
    N_val = np.size(X_train,axis=0)-N_train
    test_fold =  np.concatenate(((np.full((N_train),-1),np.full((N_val),0))))
    ps = PredefinedSplit(test_fold.tolist())

    # Regress & Predict
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    # Loop over maturities
    for i in range(Y_train.shape[1]):

        model = ElasticNetCV(cv=ps, max_iter=5000, n_jobs=-1,
                             l1_ratio=[.1, .3, .5, .7, .9],
                             random_state=42)
        model = model.fit(np.concatenate((X_train,Xexog_train),axis=1),
                          Y_train[:,i])
        Ypred[0,i]=model.predict(np.concatenate((X_test,Xexog_test),axis=1))

    return Ypred

