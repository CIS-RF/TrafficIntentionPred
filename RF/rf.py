"""
This file contains the main code of naive random forest
"""

from enum import Enum
from typing import Dict
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
#from sklearn.externals import joblib
import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np



class StateEnum(Enum):
    Inited = 0
    Fitted = 1

classifier_mapper = {'gbdt':GradientBoostingClassifier,'rf':RandomForestClassifier}

model_saved_path = r"E:\1-suyang\CIS\proj\RF\saved_model.txt"
res_saved_path = r"E:\1-suyang\CIS\proj\RF\saved_report.txt"

def config_logger():
    logger = logging.getLogger(__name__) 
    logger.setLevel(logging.DEBUG)   
    logger.info("Logger built.")
    return logger


class RandomForest():

    def __init__(self,if_load=False,classifier_type=None, train_ratio = 0.7,use_param_grid:bool=True ,parameters:Dict=None) -> None:
        
        self.logger = config_logger()
        if not if_load:
            assert classifier_type in classifier_mapper.keys(), 'classifier_type must be gbdt or rf !'
            ClassifierType = classifier_mapper[classifier_type]
            if use_param_grid:
                self._classifier = ClassifierType()
                self._grid_searcher = GridSearchCV(self._classifier,param_grid=parameters)
            else:
                self._classifier = ClassifierType(**parameters)

            self._use_param_grid = use_param_grid
            self._state = StateEnum.Inited
        else:
            self.load()
            
        self.train_ratio = train_ratio
        self._train_data = None
        self._test_data = None
        self._train_label = None
        self._test_label = None

    def set_dataset(self,data_set:np.ndarray,label:np.ndarray):
        assert data_set is not None and label is not None, " data  must not be None! "
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
        if self._state == StateEnum.Fitted:
            self.logger.info(" Refitting model.")
        if not self._use_param_grid:
            self._classifier.fit(self._train_data,self._train_label)
        else:
            self._grid_searcher.fit(self._train_data,self._train_label)
        self._state = StateEnum.Fitted

    def score(self):
        assert self._state is StateEnum.Fitted, " the classifier has not been fitted! call meth: fit() first!"
        assert self._test_data is not None and self._test_label is not None, " data must not be None! call meth: set_dataset() first."
        if not self._use_param_grid:
            return {"score":self._classifier.score(self._test_data,self._test_label)}
        else:
            cv_result = pd.DataFrame.from_dict(self._grid_searcher.cv_results_)
            best_param = self._grid_searcher.best_params_
            best_score = self._grid_searcher.best_score_
            test_pred = self._grid_searcher.predict(self._test_data)
            test_report = classification_report(y_true=self._test_label,y_pred=test_pred)
            report_dict = {"cv_result":cv_result,"best_param":best_param,"best_score":best_score,"test_report":test_report}
            with open(res_saved_path,mode='wb') as f:
                pickle.dump(report_dict,f)
            return report_dict


    def predict_class(self,x:np.ndarray):
        if self._use_param_grid:
            test_pred = self._grid_searcher.predict(x)
            return test_pred
        else:
            return self._classifier.predict(x)
    
    def predict_class_prob(self,x:np.ndarray):
        return self._classifier.predict_proba(x)
    
    def predict_class_prob_log(self,x:np.ndarray):
        return self._classifier.predict_log_proba(x)

    def reset_classifier(self,cfg:Dict):
        self._classifier = RandomForestClassifier(cfg)
        self._state = StateEnum.Inited
    

    def load(self):
        self.logger.info("Loading model from file, but without dataset.")
        with open(model_saved_path,mode='rb') as f:
            saved_dict = pickle.load(f)
            self._state = saved_dict['state']
            self._classifier = saved_dict['classifier']
            if 'grid_searcher' in saved_dict.keys():
                self._use_param_grid = True
                self._grid_searcher = saved_dict['grid_searcher']
    

    def save(self):
        self.logger.info("Saving model to file, but without dataset.")
        saved_dict = {'classifier':self._classifier,'state':self._state}
        if self._use_param_grid:
            saved_dict['grid_searcher'] = self._grid_searcher
        with open(model_saved_path,mode='wb') as f:
            pickle.dump(saved_dict,f)

