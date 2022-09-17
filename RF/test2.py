
import pickle

from rf import res_saved_path,model_saved_path

with open(res_saved_path,mode='rb') as f:
    res = pickle.load(f)
    
    print(res['test_report'])