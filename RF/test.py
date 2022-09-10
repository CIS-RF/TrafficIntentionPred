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


def test_for_rf():

    data_arr = np.loadtxt(data_path)
    label_arr = np.loadtxt(label_path)

    rf = RandomForest()
    rf.set_dataset(data_arr,label_arr)
    rf.fit()
    rf.score()




if __name__ == '__main__':
    
    
    #test()

    test_for_rf()

    
    pass