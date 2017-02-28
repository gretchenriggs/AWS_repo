import boto
import cPickle
import numpy as np

def s3_upload_files(bucketname, *args):
    """With the first string as the name of a bucket on s3, upload each individual
    file from the filepaths listed in the list of strings.
    Parameters
    ----------
    bucketname: String, List of Strings
    *args: Strings, each representing a filepath to upload.
    Returns
    -------
    None. Side effect is files will be uploaded.
    """
    access_key, secret_access_key = get_aws_access()
    conn = boto.connect_s3(access_key, secret_access_key)

    if conn.lookup(bucket_name) is None:
        bucket = conn.create_bucket(bucket_name, policy='public-read')
    else:
        bucket = conn.get_bucket(bucket_name)

    for filename in args:
            key = bucket.new_key(filename)
            key.set_contents_from_filename(filename)
            
if __name__ == '__main__':
    data_url = 'https://s3.amazonaws.com/capproj2017/X_arr_100.pkl'
    X = cPickle.load(open(data_url))
    X = np.asarray(X)

    y = 'https://s3.amazonaws.com/capproj2017/Image_Labels_125x125_1-100.txt'
