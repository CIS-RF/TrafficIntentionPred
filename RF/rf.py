"""
This file contains the main code of naive random forest
"""


from enum import Enum
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import numpy as np


class StateEnum(Enum):
    Inited = 0
    Fitted = 1


class RandomForest():

    def __init__(self, train_ratio = 0.6 ,*args,**kwargs) -> None:
        self._classifier = RandomForestClassifier(*args,**kwargs)
        self._state = StateEnum.Inited
        self.train_ratio = train_ratio

    def set_dataset(self,data_set:np.ndarray,label:np.ndarray):
        assert type(data_set) is np.ndarray and type(label) is np.ndarray, " input dataset must be a type of ndarray!"
        assert data_set.shape[0] == label.shape[0], " input data and label's first dim must match!"
        self._data_size = data_set.shape[0]
        a,b,c,d = self._process_data_set(data_set,label)
        self._train_data = a 
        self._train_label = b 
        self._test_data = c 
        self._test_label = d
    
    def _process_data_set(self,data_set,label):

        label = np.reshape(label,(-1,1))
        concat_arr = np.concatenate( (data_set,label),axis=1 )
        np.random.shuffle(concat_arr)
        train_size = int( concat_arr.shape[0] * self.train_ratio )
        train_data = concat_arr[:train_size,:-1]
        train_label = concat_arr[:train_size,-1:]
        train_label = np.squeeze(train_label)
        test_data = concat_arr[train_size:,:-1]
        test_label = concat_arr[train_size:,-1:]
        test_label = np.squeeze(test_label)

        return train_data,train_label,test_data,test_label
        
    def fit(self):
        self._classifier.fit(self._train_data,self._train_label)
        self._state = StateEnum.Fitted

    def score(self):
        assert self._state is StateEnum.Fitted, " the classifier has not been fitted! call meth: fit() first!"
        return self._classifier.score(self._test_data,self._test_label)

    def predict_class(self,x:np.ndarray):
        return self._classifier.predict(x)
    
    def predict_class_prob(self,x:np.ndarray):
        return self._classifier.predict_proba(x)
    
    def predict_class_prob_lof(self,x:np.ndarray):
        return self._classifier.predict_log_proba(x)

    def reset_classifier(self,cfg:Dict):
        self._classifier = RandomForestClassifier(cfg)
        self._state = StateEnum.Inited
    
    def load(self,path:str):
        self._classifier =  joblib.load(path)
        self._state = StateEnum.Fitted

    def save(self,path:str):
        assert self._state is StateEnum.Fitted, " the classifier has not been fitted, dump abortted."
        joblib.dump(self._classifier,path)
   

