import cPickle
import numpy as np

if __name__ == '__main__':
    # Reading false positive test images in from pickle file
    X_test_false_manmade = cPickle.load(open('X_test_false_manmade_images.pkl'))
    X_test_false_manmade = np.asarray(X_test_false_manmade)

    # Hard coding X_train_mean from X_train dataset
    X_train_mean = 0.33601385251011318
    X_test_false_manmade = (X_test_false_manmade + X_train_mean) * 255

'''' To view images:
import matplotlib.pyplot as plt

plt.imshow(X_test_false_manmade[500])
plt.show()
'''
