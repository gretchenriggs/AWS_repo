import numpy as np
from sklearn.metrics import auc

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list
    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''

    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []
    aucs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)
        aucs.append(auc(fpr,tpr))
        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, aucs, thresholds.tolist()


probabilities = model.predict_proba(X_train)[:, 1]
TPRs, FPRs, aucs, thresholds = roc_curve(probabilities, y_test)
plt.plot(FPRs, TPRs, 'b-')
# plt.plot(aucs)
plt.show()
