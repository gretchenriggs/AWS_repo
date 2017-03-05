import cPickle
import resource
import sys

# Saving out mis-classified images from test set for review.

if __name__ == '__main__':
    # Converting binary class vector y_test to 1D vector
    y_test_1d = y_test[:,1]

    X_test_false_manmade = []
    X_test_false_nature = []
    indx_false_manmade = []
    indx_false_nature = []
    for i in xrange(len(y_test_1d)):
        # Saving out False Positives (Natural images incorrectly classified as
        #   man-made.)
        if y_test_1d[i] == 0 and y_test_pred[i] == 1:
            X_test_false_manmade.append(X_test[i])
            indx_false_manmade.append(i)
        # Saving out False Negatives (Man-made images incorrectly classified as
        #   natural.)
        elif y_test_1d[i] == 1 and y_test_pred[i] == 0:
            X_test_false_nature.append(X_test[i])
            indx_false_nature.append(i)

    # Upping max recursion limit so don't run out of resources while cPickling
    #   the model.
    max_rec = 0x100000

    # May segfault without this line. 0x100 is a guess at the size of each stack frame.
    resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
    sys.setrecursionlimit(max_rec)

    # Save the False Positives (False Man-made Images) to disk
    # with open('X_test_false_manmade_images.pkl', 'wb') as pkl_fp:
    #     cPickle.dump(X_test_false_manmade, pkl_fp)
    with open('indx_false_manmade_images.pkl', 'wb') as pkl_indx_fp:
        cPickle.dump(indx_false_manmade, pkl_indx_fp)


    # Save the False Negatives (False Nature Images) to disk
    # with open('X_test_false_nature_images.pkl', 'wb') as pkl_fn:
    #     cPickle.dump(X_test_false_nature, pkl_fn)
    with open('indx_false_nature_images.pkl', 'wb') as pkl_indx_fn:
        cPickle.dump(indx_false_nature, pkl_indx_fn)

    # Save out X_test_orig & y_test_orig
    with open('X_test_orig.pkl', 'wb') as pkl_X_test:
        cPickle.dump(X_test_orig, pkl_X_test)
    with open('y_test.pkl', 'wb') as pkl_y_test:
        cPickle.dump(y_test, pkl_y_test)
