
import pickle

saved_path = r"E:\1-suyang\CIS\proj\RF\saved.txt"

with open(saved_path,mode='rb') as f:
    cv_result,best_param,best_score,test_report = pickle.load(f)
    import ipdb
    ipdb.set_trace()
    #print(test_report)