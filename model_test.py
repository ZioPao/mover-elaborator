import pickle
from itertools import cycle
from numpy import interp
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from methods_for_tests import *


walk_X = pickle.load(open('datasets_to_compile/walk.bin', 'rb'))
walk_X = np.array(walk_X)
n_samples, nx, ny = walk_X.shape
walk_X_r = walk_X.reshape((n_samples, nx*ny))
walk_y = np.zeros(n_samples)

run_X = pickle.load(open('datasets_to_compile/run.bin', 'rb'))
run_X = np.array(run_X)
n_samples, nx, ny = run_X.shape
run_X_r = run_X.reshape((n_samples, nx*ny))
run_y = np.full(n_samples, 1)

side_left_X = pickle.load(open('datasets_to_compile/side_left.bin', 'rb'))
side_left_X = np.array(side_left_X)
n_samples, nx, ny = side_left_X.shape
side_left_X_r = side_left_X.reshape((n_samples, nx*ny))
side_left_y = np.full(n_samples, 2)

side_right_X = pickle.load(open('datasets_to_compile/side_right.bin', 'rb'))
side_right_X = np.array(side_right_X)
n_samples, nx, ny = side_right_X.shape
side_right_X_r = side_right_X.reshape((n_samples, nx*ny))
side_right_y = np.full(n_samples, 3)

final_X = np.vstack((walk_X_r, run_X_r))
final_X = np.vstack((final_X, side_left_X_r))
final_X = np.vstack((final_X, side_right_X_r))
final_y = np.append(walk_y, run_y)
final_y = np.append(final_y, side_left_y)
final_y = np.append(final_y, side_right_y)

# let's mix it up boy
np.random.seed(0)
test_sub = int(0.3*len(final_y))        # 30% of the dataset
indices = np.random.permutation(len(final_X))
X_train = final_X[indices[:-test_sub]]
y_train = final_y[indices[:-test_sub]]
X_test = final_X[indices[-test_sub:]]
y_test = final_y[indices[-test_sub:]]


c_value = 24        # 24 is the best value
cv = StratifiedKFold(random_state=42, shuffle=True)

clf_svc = svm.SVC(probability=True, gamma='scale', C=c_value)        # More than 1 will get better result, 0.83 vs 0.93. But is it overfitting?
clf_svc.fit(X_train, y_train)
calibrated_clf_svc = CalibratedClassifierCV(base_estimator=clf_svc, cv="prefit")
calibrated_clf_svc.fit(X_test, y_test)

clf_knn = KNeighborsClassifier(p=1)
clf_knn.fit(X_train, y_train)
calibrated_clf_knn = CalibratedClassifierCV(base_estimator=clf_knn, cv='prefit')
calibrated_clf_knn.fit(X_test, y_test)


#pickle.dump(clf_svm, open('trained_models/mod8.bin', 'wb'))

#load old dataset
#old_dataset = [final_X, final_y]
#pickle.dump(old_dataset, open('old_dataset.bin', 'wb'))
old_dataset = pickle.load(open('old_dataset.bin', 'rb'))
X_test_diff, y_test_diff = old_dataset

print_cm(calibrated_clf_svc, "Calibrated SVC", X_test_diff, y_test_diff)
print_cm(calibrated_clf_knn, "Calibrated KNN", X_test_diff, y_test_diff)
print_cm(clf_svc, "Uncalibrated SVC", X_test_diff, y_test_diff)
print_cm(clf_knn, "Uncalibrated KNN", X_test_diff, y_test_diff)

# Doesn't work with another y_test and must be uncalibrated!
# This is totally broken
print_learning_curves(clf_svc, "SVC", clf_knn, "KNN", X_test, y_test, cv)

# param_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] p

print_validation_curves(clf_svc, 'SVC', X_test, y_test, 'C',  param_range=np.linspace(0, 50, 200))
print_validation_curves(clf_knn, 'KNN', X_test, y_test, 'n_neighbors', np.linspace(0, 50, dtype=int))
print_validation_curves(clf_knn, 'KNN', X_test, y_test, 'p', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])















# Binarize the output
final_y = label_binarize(final_y, classes=[0., 1., 2., 3.])
n_classes = final_y.shape[1]

# let's mix it up boy
np.random.seed(0)
test_sub = int(0.3*len(final_y))        # 30% of the dataset
indices = np.random.permutation(len(final_X))
X_train = final_X[indices[:-test_sub]]
y_train = final_y[indices[:-test_sub]]
#weight_train = weight[indices[:-test_sub]]

X_test = final_X[indices[-test_sub:]]
y_test = final_y[indices[-test_sub:]]


clf_knn = OneVsRestClassifier( KNeighborsClassifier(n_neighbors=5, p=1))
clf_knn.fit(X_train, y_train)
y_score = clf_knn.predict_proba(X_test)


plot_calibration_curves("KNN", 4, y_score, y_test)


clf_svc = OneVsRestClassifier(svm.SVC(probability=True, gamma='scale', C=c_value))
y_score = clf_svc.fit(X_train, y_train).decision_function(X_test)

clf_svc.decision_function(X_test)


plot_calibration_curves("SVC", n_classes, y_score, y_test)




#########################################################
def plot_calibration_curves(name, n_classes, y_score, y_test):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(name)
    plt.legend(loc="lower right")
    plt.show()

def cal_curves(name, y_test, y_prob):
    macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(
        y_test, y_prob, multi_class="ovo", average="weighted"
    )
    macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = roc_auc_score(
        y_test, y_prob, multi_class="ovr", average="weighted"
    )


    print(name)
    print(
        "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
        "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
    )
    print(
        "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
        "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
    )
    print()

