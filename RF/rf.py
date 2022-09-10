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

    def __init__(self,*args,**kwargs) -> None:
        self._classifier = RandomForestClassifier(*args,**kwargs)
        self._state = StateEnum.Inited

    def set_dataset(self,data_set:np.ndarray,label:np.ndarray):
        assert type(data_set) is np.ndarray and type(label) is np.ndarray, " input dataset must be a type of ndarray!"
        assert data_set.shape[0] == label.shape[0], " input data and label's first dim must match!"
        self._data_set = data_set
        self._data_size = data_set.shape[0]
        self._label = label

    def fit(self):
        self._classifier.fit(self._data_set,self._label)
        self._state = StateEnum.Fitted

    def score(self):
        assert self._state is StateEnum.Fitted, " the classifier has not been fitted! call meth: fit() first!"
        return self._classifier.score(self._data_set,self._label)

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
   

