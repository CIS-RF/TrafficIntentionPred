from cgi import print_arguments
from typing import Dict
import numpy as np

from rf import RandomForest

data_path = r"E:\1-suyang\CIS\proj\RF\data.txt"

label_path = r"E:\1-suyang\CIS\proj\RF\label.txt"

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


def test_for_rf(config:Dict):

    data_arr = np.loadtxt(data_path)
    label_arr = np.loadtxt(label_path)

    rf = RandomForest(**config)
    rf.set_dataset(data_arr,label_arr)
    rf.fit()
    return rf.score()



if __name__ == '__main__':
    
    rf_config = {
        'n_estimators' : 12,
        'criterion' : "gini",
        'max_depth' : 6,
        'min_samples_split' : 30,
        'min_samples_leaf' :30,
    }
    
    score = test_for_rf(rf_config)
    print(score)
    pass