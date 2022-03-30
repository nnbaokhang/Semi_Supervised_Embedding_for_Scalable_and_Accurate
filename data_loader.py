import numpy as np
import pandas as pd
from os.path import join,split
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def load_data(DATASET,DATASET_ROOT,normalize=False):

    train_data_path = join(DATASET_ROOT,DATASET,DATASET+"_TRAIN.tsv")
    train_data_raw = pd.read_csv(train_data_path,sep="\t",header=None)
    train_data = train_data_raw.values[:,1:].astype(np.float32)
    train_data[np.isnan(train_data)] = 0
    train_labels = train_data_raw.values[:,0]
    train_labels = train_labels - 1
    #train_data = np.expand_dims(np.expand_dims(train_data,axis=-1),axis=1)

    test_data_path = join(DATASET_ROOT,DATASET,DATASET+"_TEST.tsv")
    test_data_raw = pd.read_csv(test_data_path,sep="\t",header=None)
    test_data = test_data_raw.values[:,1:].astype(np.float32)
    test_data[np.isnan(test_data)] = 0
    test_labels = test_data_raw.values[:,0]
    test_labels = test_labels - 1

    # UWAVE SPECIFIC
    if DATASET == "UWaveGestureLibraryAll":
        train_data = train_data.reshape(train_data.shape[0],3,-1)
        test_data = test_data.reshape(test_data.shape[0],3,-1)
        # Skip 930
        test_data = np.concatenate([test_data[:930],test_data[931:]])
        test_labels = np.concatenate([test_labels[:930],test_labels[931:]])

    train_data = np.concatenate([train_data,test_data],axis=0)
    train_labels = np.concatenate([train_labels,test_labels],axis=0)

    if normalize:
        td_max = train_data.max(axis=-1, keepdims=True)
        td_min = train_data.min(axis=-1, keepdims=True)
        train_data = (2*(train_data - td_min)/(td_max-td_min))-1

    if DATASET == "UWaveGestureLibraryAll":
        train_data = np.transpose(train_data,(0,2,1))
        test_data = np.transpose(test_data,(0,2,1))
    if train_data.ndim == 2:
        train_data = np.expand_dims(train_data, axis=-1)
        test_data = np.expand_dims(test_data, axis=-1)

    return train_data,train_labels,None


def custom_load_data(dataset_name):
    X_train_ = pd.read_csv(dataset_name)
    X_train = X_train_
    X_train = X_train.T
    import ast
    # Loop through the column and take out the first row, then concatenate all the time series as 2d array
    # Dataset
    r, c = X_train_.T.shape
    sufficient_rating_arr = []
    for j in range(c):
        data = []
        sufficient_rating = []
        try:
            for i in X_train.iloc[1:][j]:
                data.append(np.delete(np.array(ast.literal_eval(i)), 3))  # Remove year
                sufficient_rating.append(data[-1][1])
            sufficient_rating_arr.append(sufficient_rating)
        except ValueError:
            print(ValueError)
            pass
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(sufficient_rating_arr)
    return X_scaled
