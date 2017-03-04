import cPickle
import numpy as np

if __name__ == '__main__':
    # Reading false negative test images in from pickle file
    X_test_false_nature = cPickle.load(open('X_test_false_nature_images.pkl'))
    X_test_false_nature = np.asarray(X_test_false_nature)

    # Hard coding X_train_mean from X_train dataset
    X_train_mean = 0.33601385251011318
    X_test_false_nature = (X_test_false_nature + X_train_mean) * 255

'''' To view images:
import matplotlib.pyplot as plt

plt.imshow(X_test_false_manmade[500])
plt.show()
'''
