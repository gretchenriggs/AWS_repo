import cPickle
from sklearn.model_selection import train_test_split
import numpy as np

# Specifically to read data from AWS S3 bucket
import boto
import os
access_key = os.environ['AWS_ACCESS_KEY_ID']
secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']


def get_X_from_bucket(file):
    ''' Download feature matrix X from S3 bucket
        Input: File name containing pickled feature matrix X
        Output: Feature matrix X as an array
    '''
    # Create connection to s3 and connect to the bucket
    conn = boto.connect_s3(access_key, secret_access_key)

    # Setting bucket of interest
    bucket = conn.get_bucket('capproj2017')

    # Setting specific file of interest to file_key
    file_key = bucket.get_key(file)

    file_key.get_contents_to_filename(file)

    # Unpickling and returning feature matrix X as an array
    X = cPickle.load(open(file))
    X = np.asarray(X)
    return X

def train_test(X, y, nb_classes, test_percent=0.20):
    ''' Split the X, y datasets into training & test sets based on
               selected percentage (optional)
        Input: X feature, array
               y label, array
        Output: X_train & X_test, arrays
                y_train & y_test, arrays, now converted to binary class
                    matrices
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                              test_size = test_percent, random_state=42)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Load in pickled 20100 124x124x3 images from AWS S3 and labels (0, 1)
    features1 = 'X_arr_6700.pkl'
    features2 = 'X_arr_6700_deg0mir.pkl'
    features3 = 'X_arr_6700_deg90.pkl'
    features4 = 'X_arr_6700_deg90mir.pkl'
    features5 = 'X_arr_6700_deg180.pkl'
    features6 = 'X_arr_6700_deg180mir.pkl'
    features7 = 'X_arr_6700_deg270.pkl'
    features8 = 'X_arr_6700_deg270mir.pkl'
    labels = 'Image_Labels_125x125_53600.txt'
    X1 = get_X_from_bucket(features1)
    X2 = get_X_from_bucket(features2)
    X3 = get_X_from_bucket(features3)
    X4 = get_X_from_bucket(features4)
    X5 = get_X_from_bucket(features5)
    X6 = get_X_from_bucket(features6)
    X7 = get_X_from_bucket(features7)
    X8 = get_X_from_bucket(features8)
    X = np.concatenate([X1, X2, X3, X4, X5, X6, X7, X8], axis=0)
    y = np.loadtxt(labels, dtype=int)


    nb_classes = 2
    # Train/test split for cross validation
    test_percent = 0.20
    X_train, X_test, y_train, y_test = \
                            train_test(X, y, nb_classes, test_percent)
    index_fp = cPickle.load(open('index_false_manmade_images.pkl'))

    index_fn = cPickle.load(open('index_false_nature_images.pkl'))

    X_test_false_manmade = []
    X_test_false_nature = []
    for i in xrange(len(X_test)):
        if i in index_fp:
            X_test_false_manmade.append(X_test[i])
        elif i in index_fn:
            X_test_false_nature.append(X_test[i])

    # Save the False Positives (False Man-made Images) to disk
    with open('X_test_false_manmade_images.pkl', 'wb') as pkl_fp:
        cPickle.dump(X_test_false_manmade, pkl_fp)

    # Save the False Negatives (False Nature Images) to disk
    with open('X_test_false_nature_images.pkl', 'wb') as pkl_fn:
        cPickle.dump(X_test_false_nature, pkl_fn)
