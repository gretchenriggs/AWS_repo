import cPickle
import numpy as np
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
    # bucket = conn.get_bucket('capproj2017')
    # b = conn.get_bucket(bucket)
    #
    # # Outputting all bucket names to variable
    # all_buckets = [b.name for b in conn.get_all_buckets()]

    # Setting bucket of interest
    bucket = conn.get_bucket('capproj2017')

    # Setting specific file of interest to file_key
    file_key = bucket.get_key(file)

    # Unpickling and returning feature matrix X as an array
    X = cPickle.load(open(file))
    X = np.asarray(X)
    return X


if __name__ == '__main__':
    features = 'X_arr_6700.pkl'
    labels = 'Image_Labels_125x125_6700.txt'
    X = get_X_from_bucket(features)
    y = np.loadtxt(labels, dtype=int)
