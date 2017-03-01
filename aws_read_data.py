import cPickle
import numpy as np
import boto
import os
access_key = os.environ['AWS_ACCESS_KEY']
secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

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

# Create connection to s3 and connect to the bucket
conn = boto.connect_s3(access_key, secret_access_key)
b = conn.get_bucket(bucket)
all_buckets = [b.name for b in conn.get_all_buckets()]

bucket = 'capproj2017'

local_file = '/tmp/X_arr_100.pkl'
bucket.get_key(aws_app_assets + "X_arr_100.pkl") \
      .get_contents_to_filename(local_file)
clf = joblib.load(local_file)
os.remove(local_file)


if conn.lookup(bucket) is None:
    b = conn.create_bucket(bucket, policy='public-read')
else:
    b = conn.get_bucket(bucket)

file_object = b.new_key('X_arr_100.pkl')
file_object.set_contents_from_string('X_arr_100.pkl, policy='public-read')

key.get_contents_to_filename(os.path.join(path, key.name))

if __name__ == '__main__':
    data_url = 'X_arr_100.pkl'
    X = cPickle.load(open(data_url))
    X = np.asarray(X)

    y = 'https://s3.amazonaws.com/capproj2017/Image_Labels_125x125_1-100.txt'
