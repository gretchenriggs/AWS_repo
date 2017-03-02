''' Running CNN to classify satellite images as containing only natural
    objects (0) or containing some man-made objects (1).
    Using Theano, with Tensorflow image_dim_ordering :
    (# images, # rows, # cols, # channels)
    (# images, 124, 124, 3) for the X_train images below
'''
import cPickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

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


# def data_load(X, labels):
#     ''' Load features and labels from pickle and text files, respectively
#         Input: cPickle file of features, with RGB values extracted and
#                    centered around 0
#                text file of labels
#         Output: X and y feature and label arrays
#     '''
#     X = cPickle.load(open(features))
#     X = np.asarray(X)
#     y = np.loadtxt(labels, dtype=int)
#     return y


def preproc(X_train, X_test):
    ''' Set pixel values to be between 0 and 1 and center them around zero.
        Input: X feature arrays
        Output: X feature arrays, standardized and centered around zero.
    '''
    # Standardizing pixel values to be between 0 and 1
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    X_train /= 255.0
    X_test /= 255.0

    # Zero-center the data (important), in steps due to memory error
    # Compute mean only on Training data to prevent leakage of info into
    #   Test set
    mean_X_train = np.mean(X_train)
    for i in range(len(X_train)):
        X_train[i] = X_train[i] - mean_X_train
    for i in range(len(X_test)):
        X_test[i] = X_test[i] - mean_X_train
    return X_train, X_test


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

    # Settting pixel values between 0-1 and centering around zero
    X_train, X_test = preproc(X_train, X_test)

    # Convert class vectors to binary class matrices.
    y_train_orig = y_train.copy()
    y_test_orig = y_test.copy()
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, y_train, y_test, y_train_orig, y_test_orig


def keras_inp_prep(X_train, X_test, img_dep, img_rows, img_cols, \
                   dim_ordering='tf'):
    ''' Reshape feature matrices X_train, X_test to be compatible with
            neural network (NN) expected input_shape
        if dim_ordering is th (Theano), NN expects:
            (# images, # channels, # rows, # cols)
        if dim_ordering is tf (TensorFlow), NN expects:
            (# images, # rows, # cols, # channels)
        The tf and th image_dim_ordering is first set in the
            ~/keras/keras.json file.  Make sure it matches the
            K.set_image_dim_ordering at the beginning of this script.

        Input: X_train, X_test, arrays
               img_dep, img_rows, img_cols, integers
               dim_ordering, string
        Output: X_train, X_test, arrays
                input_shape, tuple
    '''
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], img_dep, img_rows, \
                                  img_cols)
        X_test = X_test.reshape(X_test.shape[0], img_dep, img_rows, \
                                img_cols)
        input_shape = (img_dep, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, \
                                  img_dep)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, \
                                img_dep)
        input_shape = (img_rows, img_cols, img_dep)
    return X_train, X_test, input_shape


def cnn(X_train, y_train, X_test, y_test, kernel_size, pool_size,\
        batch_size, nb_classes, nb_epoch):
    ''' This is the Convolutional Neural Net architecture, which is the
            workhorse in this python script, for classifying the images
            as natural (0) or containing man-made images (1).
        Input: X_train, y_train, training data, arrays
               X_test, y_test, cross validation data, arrays
               kernel_size, integer (filter size)
               pool_size, tuple of integers (desampling values)
               batch_size, integer (# of images to process at a time)
               nb_epoch, integer (# of iterations of fwd/back propagation
                                  to run CNN through)
        Output: model, Sequential NN object
                score, accuracy of prediction on test data
                score_train, accuracy of prediction on training data
    '''
    model = Sequential()

    model.add(Convolution2D(32, kernel_size, kernel_size, \
                    border_mode='valid', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, kernel_size, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(64, kernel_size, kernel_size, \
                            border_mode='valid'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(64, kernel_size, kernel_size))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # During fit process, you can watch train and test error simultaneously
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score[1]) # this is the one we care about

    score_train = model.evaluate(X_train, y_train, verbose=0)
    print('Train accuracy:', score_train[1])

    return model, score, score_train


def model_performance(model, X_train, X_test, y_train, y_test):
    ''' Compute accuracy, precision, recall, F1-Score.
        Output the predicted y_test & y_train values.
        Output the probabilities for the predicted y_test & y_train values
        Output confusion matrix for y_test
        Input: model, Sequential object
               X_train, array
               y_train, array
               X_test, array
               y_test, array
        Output: y_train_pred, array
                y_test_pred, array
                y_train_pred_proba, array
                y_test_pred_probab, array
                conf_matrix, array
    '''


    # Predictions on Test and Train datasets
    y_test_pred = model.predict_classes(X_test)
    y_train_pred = model.predict_classes(X_train)

    # Predictions probability outputs for Test & Train datasets
    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)

    # Converting y_test back to 1-D array for confustion matrix computation
    y_test_1d = y_test[:,1]

    # Computing confusion matrix for Test datasets
    conf_matrix = confusion_matrix(y_test_1d, y_test_pred)
    print "Confusion Matrix: \n", conf_matrix

    return y_train_pred, y_test_pred, y_train_pred_proba, \
           y_test_pred_proba, conf_matrix


def standard_confusion_matrix(y_test, y_test_pred):
    ''' Computing Confusion Matrix for CNN model, formatting in
            standard setup.
        Input:  y_true and y_predict values (array-like)
        Output: confusion matrix            (array_like)
    '''
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])


if __name__ == '__main__':
    # Load in pickled 20100 124x124x3 images from AWS S3 and labels (0, 1)
    features1 = 'X_arr_6700.pkl'
    features2 = 'X_arr_6700_deg0mir.pkl'
    labels = 'Image_Labels_125x125_13400.txt'
    X1 = get_X_from_bucket(features1)
    X2 = get_X_from_bucket(features2)
    X = np.concatenate([X1, X2], axis=0)
    y = np.loadtxt(labels, dtype=int)
    # X, y = data_load(X, labels)

    # Setting up basic parameters needed for neural network
    batch_size = 32
    nb_classes = 2
    nb_epoch = 20
    kernel_size = 3
    pool_size = (2, 2)

    # Train/test split for cross validation
    test_percent = 0.20
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = \
                            train_test(X, y, nb_classes, test_percent)

    # input image dimensions - 124x124x3 for input RGB Satellite Images
    img_rows, img_cols, img_dep = X_train.shape[1], X_train.shape[2], \
                                  X_train.shape[3]

    # Prep input for Keras.
    # For Tensorflow dim_ordering (tf), (# images, # rows, # cols, # chans)
    # For Theano dim ordering (th), (# images, # chans, # rows, # cols)
    # Used Theano backend, tf dim_ordering for this project
    X_train, X_test, input_shape = keras_inp_prep(X_train, X_test,\
                       img_dep, img_rows, img_cols, dim_ordering='tf')

    # Run Convolutional Neural Net
    model, score, score_train = cnn(X_train, y_train, X_test, y_test,\
                        kernel_size, pool_size, batch_size, nb_classes,\
                        nb_epoch)

    # Evaluating CNN Model performance
    y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, \
       conf_matrix = model_performance(model, X_train, X_test, y_train, y_test)
