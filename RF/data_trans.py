from heapq import merge
import pandas as pd
from pandas import DataFrame
import numpy as np

from sklearn.preprocessing import ( scale,
                                    MinMaxScaler,
                                    MaxAbsScaler,
                                    maxabs_scale,
                                    minmax_scale,
                                    StandardScaler,
                                    quantile_transform,
                                    normalize
                                     )

# left_path = r"E:\1-suyang\CIS\proj\RF\lane_l.csv"
# right_path = r"E:\1-suyang\CIS\proj\RF\lane_r.csv"
# needed_entriess = ['v_Vel_X','v_Vel_Y','v_Acc_X','v_Acc_Y','Distance_l','Distance_r']
# left_frame = pd.read_csv(left_path)
# right_frame = pd.read_csv(right_path)
# left_frame = left_frame[needed_entriess]
# right_frame = right_frame[needed_entriess]
# left_frame['label'] = 1
# right_frame['label'] = 0
# merged = pd.concat([left_frame,right_frame])
# #print(len(merged[merged['label']==1]))
# save_path = r"E:\1-suyang\CIS\proj\RF\merged.csv"
# merged.to_csv(save_path)

    

# file_path = r"E:\1-suyang\CIS\proj\RF\merged.csv"
# data_path = r"E:\1-suyang\CIS\proj\RF\data.txt"
# merged_df = pd.read_csv(file_path)
# #print(merged_df.columns)
# merged_df.drop(columns=["label"],inplace=True)
# print(merged_df.columns)
# merged_arr = merged_df.values
# print(merged_arr.shape)
# merged_arr = minmax_scale(merged_arr)
# np.savetxt(data_path,merged_arr)
# #print(type(label_arr),' ',label_arr.shape)

new_data_path = r"E:\1-suyang\CIS\proj\RF\df_new.csv"
file_path = r"E:\1-suyang\CIS\proj\RF\df_1m.csv"
df = pd.read_csv(file_path)




# if __name__ == '__main__':

    
    
#     pass