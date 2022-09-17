from cgi import print_arguments
from typing import Dict
import numpy as np

from rf import RandomForest
import pickle

data_path = r"E:\1-suyang\CIS\proj\RF\data\data.txt"
label_path = r"E:\1-suyang\CIS\proj\RF\data\label.txt"


def test():

    data_arr = np.loadtxt(data_path)
    label_arr = np.loadtxt(label_path)

    label_arr = np.reshape(label_arr,(-1,1))
    #print(label_arr.shape)
    test_arr = np.concatenate( (data_arr,label_arr),axis=1 )
    #print(test_arr.shape)
    np.random.shuffle(test_arr)

    label = test_arr[:200,:-1]
    print(label.shape)

    pass


def test_for_rf(total_config:Dict):

    data_arr = np.loadtxt(data_path)
    label_arr = np.loadtxt(label_path)

    rf = RandomForest(**total_config)
    rf.set_dataset(data_arr,label_arr)
    rf.fit() #if load=True and just want to simply test, this step can be omitted.
    res = rf.score()
    rf.save()
    return res



if __name__ == '__main__':

    base_config = {
        'n_estimators' : 20,
        #'criterion' : ["gini", "entropy", "log_loss"],
        'criterion' : "gini",
        'max_depth' : 8,
        'min_samples_split' : 15,
        'min_samples_leaf' : 15,
    }
    
    rf_config = {
        'n_estimators' : range(20,120,10),
        #'criterion' : ["gini", "entropy", "log_loss"],
        'criterion' : ["gini"],
        'max_depth' : range(4,12,2),
        'min_samples_split' : range(10,30,5),
        'min_samples_leaf' : range(5,20,5),
    }

    gbdt_config = {
        'n_estimators' : range(20,120,20),
        'learning_rate': [0.1,0.05,0.2],
        'max_depth' : range(4,12,4),
        'min_samples_split' : range(10,30,5),
        'min_samples_leaf' : range(5,20,5),
    }

    use_grids = True
    clsfer_type = 'gbdt'
    load_file = None

    total_cfg = {
        'parameters': gbdt_config,
        'use_param_grid':  use_grids,
        'classifier_type':  clsfer_type,
        'load_file':load_file
    }
    
    score = test_for_rf(total_cfg)
    print(score)
    
    pass