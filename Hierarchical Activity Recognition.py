# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:56:34 2019

@author: 14342
"""
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import glob
import os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# - data passed should have all the labels (parent and child level) ready

def add_label(L_in):
    """Adding the "label:" to the name of the labels so that we can call them
       from the dtaframe of the dataset
       
       ex. ['Lying_down','Talking']  ->  ['label:LYING_DOWN','label:TALKING']
       """
       
    L_out = []
    
    for i in range(len(L_in)):
        L_out.append('label:'+ L_in[i].upper())
    
    return(L_out)

def drop_label(L_in):
    """Rmoving the "label:" from the name of the labels so that we can use them
       for a better comprehension
       
       ex. ['label:LYING_DOWN','label:TALKING']  ->  ['Lying_down','Talking']
       """
    L_out = []
    
    for i in range(len(L_in)):
        L_out.append(L_in[i][6:].title())
        
    return(L_out)

def readdata_csv(data_dir):
    """This function gets the directory of the datasets and returns the dataset
    containing information of all 60 users
    
    Input:
        data_dir[string]: holds the directory of all the csv files (60)
        
    Output:
        grand_dataset[dict]: a dictionary of all the users' data. The format is:
            grand_dataset:{'uuid1': dataframe of csv file of the user 1
                           'uuid2': dataframe of csv file of the user 2
                           'uuid3': dataframe of csv file of the user 3
                           ...}
    
    """
    length_uuids = 36 # number of characters for each uuid
    data_list = glob.glob(os.path.join(os.getcwd(), data_dir, "*.csv"))
    # grand_dataset is a dict. that holds the uuids and correspondong datast
    grand_dataset = {}
    lengthOFdataset = 0
    for i in range(len(data_list)):
#    for i in range(5):
        # dismantles the file name and picks only uuids (first 36 characters)
        uuid = os.path.basename(data_list[i])[:length_uuids]
        dataset_ith = pd.read_csv(data_list[i])
        print(i,dataset_ith.shape)
        lengthOFdataset += len(dataset_ith)
        grand_dataset[uuid] = dataset_ith
    print(lengthOFdataset)
    return(grand_dataset)

def data_cleaner(feature_set_range, parent_labels, child_labels, dataset):
    """This function cleans the data given to it and produced X and y for the
    parent and child levels that can be used to train classifiers.
    
    Inpout:
        featur_set_range[range]: the range of the features to be used for our models
        parent_labels[list]: list of the parent labels
        child_labels[list of list]: a list of list of the child labels. Each element
                                    is the list of one parent branch
        dataset[pd dataframe]: the dataframe to extract X and y from
    
    Output:
        X_parent[pd dataframe]: features of the parent level model
        y_parent[pd dataframe]: response of the parent level model
        X_child[list(pd dataframe)]: list of the features for the child level models
        y_child[list(pd dataframe)]: list of the responses for the child level models
    """
    
    num_parents = len(parent_labels)       # Number of the classes at parent level
    assert len(parent_labels) == len(child_labels)
    
    # Cutting out the features and labels from the entire dataset
    features = dataset.iloc[:,feature_set_range]
    labels = dataset[parent_labels]
    
    # Consider eliminating the for loops
    for i in range(num_parents):
        for j in range(len(child_labels[i])):
            labels = pd.concat([labels,dataset[child_labels[i][j]]], axis=1)
    
    # We attach the "features" and "labels" to get the raw data
    raw_data = pd.concat([features,labels],axis=1)
    raw_data = raw_data.dropna()
    
    # We are making sure that we have samples that belong to exactly one class
    # of parent labels at the same time. As the parent labels are chosen so that
    # to be completely exclusive, we expect to get none of such samples
    
    # Parent level:
    raw_data['Parent_Belonging'] = 0   # Number of parent classes a sample belongs to
    raw_data['Child_Belonging'] = 0    # Number of child classes a sample belongs to
    
    for i in range(num_parents):
        raw_data['Parent_Belonging'] += (raw_data[parent_labels[i]] == 1)*1
    
    #Child level:
    for i in range(num_parents):
        for j in range(len(child_labels[i])):
            raw_data['Child_Belonging'] += (raw_data[child_labels[i][j]] == 1)*1
    
    raw_data = raw_data[raw_data['Parent_Belonging'] == 1]
    raw_data = raw_data[raw_data['Child_Belonging'] == 1]
    
    print('Number of samples remaining after removing missing features and labels: '\
          ,len(raw_data))
    
    # Adding the parent label column to the data
    raw_data['Parent_label_index'] = 0    # The index of the parent class that each sample belongs to
    raw_data['Child_label_index'] = 0     # The index of the child class that each sample belongs to
    
    for i in range(num_parents):
        raw_data['Parent_label_index'] += (i+1)*(1*(raw_data[parent_labels[i]] == 1))
    
    for i in range(num_parents):
        for j in range(len(child_labels[i])):
            raw_data['Child_label_index'] += (j+1)*(1*(raw_data[child_labels[i][j]] == 1))
    
    raw_data['Parent_label_index'] -= 1
    raw_data['Child_label_index'] -= 1
    
    # Collecting X and y for the training phase of the classifiers
    X_parent = raw_data.iloc[:,feature_set_range]
    X_parent = preprocessing.scale(X_parent, axis=1)
    y_parent = raw_data['Parent_label_index']
    
    X_child = []
    y_child = []
    for i in range(num_parents):
        dataset_ith_parent = raw_data[raw_data['Parent_label_index'] == i]
        X_child.append(dataset_ith_parent.iloc[:,feature_set_range])
        X_child[i] = preprocessing.scale(X_child[i], axis=1)
        y_child.append(dataset_ith_parent['Child_label_index'])
    
    return(X_parent, y_parent, X_child, y_child)

if __name__ == '__main__':
    
    data_dir = "C:\\Mehrdad\\DARPA-WASH\\datasets\\"
    cvdir = "C:\\Mehrdad\\DARPA-WASH\\cv5Folds\\cv_5_folds\\"
    
    dataset_uuids = readdata_csv(data_dir) #reading all data and storing in "dataset" a DF
    
    uuids = list(dataset_uuids.keys())
    
    dataset = dataset_uuids[uuids[0]]
    
    for i in range(1,len(uuids)):
        dataset = pd.concat([dataset,dataset_uuids[uuids[i]]],axis=0)
    
    feature_set_range = range(1,84)
    
    ##########################################################################
    #-----------------| Stationary vs NonStationary |------------------------#
    ##########################################################################
    
    #We don't have these labels so we add them by our own Based on the child labels defined
    parent_labels = ['Stationary','NonStationary']
    
    child_labels = [['label:OR_standing','label:SITTING','label:LYING_DOWN'],\
                    ['label:FIX_running','label:FIX_walking','label:BICYCLING']]
    
    feat = dataset.iloc[:,feature_set_range]
    response = pd.DataFrame()
    for i in range(len(parent_labels)):
        for j in range(len(child_labels[i])):
            response = pd.concat([response,dataset[child_labels[i][j]]], axis=1)

    data_to_model = pd.concat([feat,response],axis=1)
    data_to_model = data_to_model.dropna()
    
    data_to_model['Stationary'] = np.logical_or(data_to_model['label:OR_standing'],data_to_model['label:SITTING'])
    data_to_model['Stationary'] = np.logical_or(data_to_model['Stationary'],data_to_model['label:LYING_DOWN'])*1
    
    data_to_model['NonStationary'] = np.logical_or(data_to_model['label:FIX_running'],data_to_model['label:FIX_walking'])
    data_to_model['NonStationary'] = np.logical_or(data_to_model['NonStationary'],data_to_model['label:BICYCLING'])*1
       
    feature_set_range = range(83)
    X_parent, y_parent, X_child, y_child = data_cleaner(feature_set_range, parent_labels, child_labels, data_to_model)
    
    ###########################################################################
    #-------------------------| Indoor vs Outdoor |---------------------------#
    ###########################################################################
    
#    parent_labels = ['label:OR_indoors','label:OR_outside']
#    
#    child_labels = [['label:IN_A_MEETING','label:IN_CLASS','label:AT_HOME'],\
#                    ['label:FIX_running','label:FIX_walking','label:BICYCLING']]
#    
#    feat = dataset.iloc[:,feature_set_range]
#    response = pd.DataFrame()
#    for i in range(len(parent_labels)):
#        for j in range(len(child_labels[i])):
#            response = pd.concat([response,dataset[child_labels[i][j]]], axis=1)
#
#    dataset = pd.concat([feat,response],axis=1)
#    dataset = dataset.dropna()
#    
#    dataset['Stationary'] = np.logical_or(dataset['label:OR_standing'],dataset['label:SITTING'])
#    dataset['Stationary'] = np.logical_or(dataset['Stationary'],dataset['label:LYING_DOWN'])*1
#    
#    dataset['NonStationary'] = np.logical_or(dataset['label:FIX_running'],dataset['label:FIX_walking'])
#    dataset['NonStationary'] = np.logical_or(dataset['NonStationary'],dataset['label:BICYCLING'])*1
#       
#    feature_set_range = range(83)
#    X_parent, y_parent, X_child, y_child = data_cleaner(feature_set_range, parent_labels, child_labels, dataset)
    
    
    ###########################################################################
    #----------------------------| Phone Position |---------------------------#
    ###########################################################################
    
    parent_labels = ['label:PHONE_IN_BAG','label:PHONE_IN_HAND','label:PHONE_IN_POCKET',\
                     'label:PHONE_ON_TABLE']
    
    child_labels = [['label:PHONE_IN_BAG'],['label:PHONE_IN_HAND'],['label:PHONE_IN_POCKET'],\
                    ['label:PHONE_ON_TABLE']]
    
    feat = dataset.iloc[:,feature_set_range]
    response = pd.DataFrame()
    for i in range(len(parent_labels)):
        for j in range(len(child_labels[i])):
            response = pd.concat([response,dataset[child_labels[i][j]]], axis=1)

    dataset = pd.concat([feat,response],axis=1)
    dataset = dataset.dropna()
    
#    dataset['Stationary'] = np.logical_or(dataset['label:OR_standing'],dataset['label:SITTING'])
#    dataset['Stationary'] = np.logical_or(dataset['Stationary'],dataset['label:LYING_DOWN'])*1
#    
#    dataset['NonStationary'] = np.logical_or(dataset['label:FIX_running'],dataset['label:FIX_walking'])
#    dataset['NonStationary'] = np.logical_or(dataset['NonStationary'],dataset['label:BICYCLING'])*1
       
    feature_set_range = range(83)
    X_parent, y_parent, X_child, y_child = data_cleaner(feature_set_range, parent_labels, child_labels, dataset)

    
    ###########################################################################
    #------------------------------| DNN |------------------------------------#
    ###########################################################################
    
    X_parent_train, X_parent_test, y_parent_train, y_parent_test = \
    train_test_split(X_parent, y_parent, test_size=0.3)
    
    clf_parent = Sequential()
    clf_parent.add(Dense(1024, input_dim=len(feature_set_range), activation='relu'))
    clf_parent.add(Dense(512, activation='relu'))
    clf_parent.add(Dense(64, activation='relu'))
    clf_parent.add(Dense(1, activation='sigmoid'))
    
    clf_parent.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    clf_parent.fit(X_parent_train, y_parent_train, batch_size=100, \
                   validation_data=(X_parent_test, y_parent_test.values), \
                   epochs=10, class_weight='balanced')
    
    y_parent_pred = clf_parent.predict(X_parent_test)
    
    #confusion_matrix = confusion_matrix(y_parent_test.values, y_parent_pred.round)
    
    df_cm = pd.DataFrame(confusion_matrix, range(2),range(2))
#    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    f1_score(y_parent_test.values, y_parent_pred.round(), average='macro')
    





        
        
    
