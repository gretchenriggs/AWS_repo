import cPickle
import numpy as np

if __name__ == '__main__':
    data_url = 'https://s3.amazonaws.com/capproj2017/X_arr_100.pkl'
    X = cPickle.load(open(data_url))
    X = np.asarray(X)

    y = 'https://s3.amazonaws.com/capproj2017/Image_Labels_125x125_1-100.txt'
