

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:18:12 2019

@author: 14342
"""
import pandas as pd
import numpy as np
import glob
import os

def get_filepath(file_dir,uuid):
    """This function gets the uuid of a subject and returns the file path for
    csv file of the subject
    
    Input:
        file_dir[string]: holds the directory of the desired file
        uuid[string]: 32 character string holding uuid
    
    Output:
        filepath[string]: a string of the address of the file that can be read easily
    
    """
    filename = '{}.features_labels.csv'.format(uuid)
    filepath = os.path.join(file_dir, filename)
    return(filepath)

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


def train_test_split_cv(test_fold,num_folds,fold_dir,grand_dataset):
    """This function takes the number of test fold (ranging from 0 to 4) and
    number of folds (in this case 5) and directory where the folds' uuids are
    and the dataset, and returns train and test datasets
    
    Input:
        test_fold_idx[integer]: an integer indicating the index of the test fold
        fold_dir[string]: holds the directory in which the folds' uuids are
        grand_dataset[dict]: a dictionary of all users' data. (essentially the
                             output of readdata_csv())
    Output:
        train_dataset[pandas.dataframe]: dataframe of the train dataset
        test_dataset[pandas.dataframe]: dataframe of the test dataset
    
    """
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()
    folds_uuids = get_folds_uuids(fold_dir)
    
    # Dividing the folds uuids into train and test (the L denotes they are still lists)
    test_uuids_L = [folds_uuids[test_fold]]
    del(folds_uuids[test_fold])
    train_uuids_L = folds_uuids
    
    # Transforming the list of arrays of uuids into a single uuids np.array
    test_uuids = np.vstack(test_uuids_L)
    train_uuids = np.vstack(train_uuids_L)
    
    # Now collecting the test and train dataset using concatenating
    for i in train_uuids:
        train_dataset = pd.concat([train_dataset,grand_dataset[i[0]]])
    
    for j in test_uuids:
        test_dataset = pd.concat([test_dataset,grand_dataset[j[0]]])
        
    return(train_dataset,test_dataset)


def get_folds_uuids(fold_dir):
    """
    The function gets the directory where the the folds text files are located
    and returns a list of five np.arrays in each of them the uuids of the
    corresponding fold are stored.
    
    Input:
        fold_dir[string]: holds the directory in which folds are
    
    Output:
        folds_uuids[list]: a list of numpy arrays. Each array holds the uuids
                    in that fold. ex.
                    folds_uuids = [('uuid1','uuid2',...,'uuid12'),
                                   ('uuid13','uuid14',...,'uuid24'),
                                   ...,
                                   ('uuid49','uuid50',...,'uuid60')]
    """
    num_folds = 5
    # folds_uuids is gonna be a list of np.arrays. each array is a set of uuids
    folds_uuids = [0,1,2,3,4]
    # This loop reads all 5 test folds (iphone and android) and stores uuids
    for i in range(0,num_folds):
        filename = 'fold_{}_test_android_uuids.txt'.format(i)
        filepath = os.path.join(fold_dir, filename)
        # aux1 is the uuids of ith test fold for "android"
        aux1 = pd.read_csv(filepath,header=None,delimiter='\n')
        aux1 = aux1.values
        
        filename = 'fold_%s_test_iphone_uuids.txt' %i
        filepath = os.path.join(fold_dir, filename)
        # aux2 is the uuids of ith test fold for "iphone"
        aux2 = pd.read_csv(filepath,header=None,delimiter='\n')
        aux2 = aux2.values
        
        # Then we concatenate them
        folds_uuids[i] = np.concatenate((aux1,aux2),axis=0)
        
    return(folds_uuids)


def sensors():
    """This function sets the ranges of the various sensors"""
    Sensor = {}
    Sensor['Acc'] = list(range(1,27))
    Sensor['Gyro'] = list(range(27,53))
    Sensor['Mag'] = list(range(53,84))
    Sensor['W_acc'] = list(range(84,130))
    Sensor['Compass'] = list(range(130,139))
    Sensor['Loc'] = list(range(139,156))
    Sensor['Aud'] = list(range(156,182))
    Sensor['AP'] = list(range(182,184))
    Sensor['PS'] = list(np.append(range(184,210),range(218,226)))
    return(Sensor)


def activities():
    activity = {}
    activity['label:LYING_DOWN'] = 226
    activity['label:SITTING'] = 227
    activity['label:FIX_walking'] = 228
    activity['label:FIX_running'] = 229
    activity['label:BICYCLING'] = 230
    activity['label:SLEEPING'] = 231
    activity['label:OR_indoors'] = 236
    activity['label:OR_outside'] = 237
    activity['label:IN_A_CAR'] = 238
    activity['label:ON_A_BUS'] = 239
    activity['label:DRIVE_-_I_M_THE_DRIVER'] = 240
    activity['label:DRIVE_-_I_M_A_PASSENGER'] = 241
    activity['label:PHONE_IN_POCKET'] = 244
    activity['label:PHONE_IN_HAND'] = 272
    activity['label:PHONE_IN_BAG'] = 273
    activity['label:PHONE_ON_TABLE'] = 274
    return(activity)


