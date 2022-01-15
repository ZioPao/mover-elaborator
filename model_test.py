import pickle
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
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

#load old dataset
old_dataset = pickle.load(open('old_dataset.bin', 'rb'))
X_test_diff, y_test_diff = old_dataset


c_value = 24        # 24 is the best value
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

clf_svc = svm.SVC(probability=True, gamma='scale', C=c_value)        # More than 1 will get better result, 0.83 vs 0.93. But is it overfitting?
clf_knn = KNeighborsClassifier(p=1)
clf_svc.fit(X_train, y_train)
clf_knn.fit(X_train, y_train)

# Calibration method 1
cal_svc_1 = CalibratedClassifierCV(
    base_estimator=svm.SVC(probability=True, gamma='scale', C=c_value).fit(X_train, y_train), cv='prefit')
cal_svc_1.fit(X_test, y_test)

cal_knn_1 = CalibratedClassifierCV(base_estimator=KNeighborsClassifier(p=1).fit(X_train, y_train).fit(X_train, y_train),
                                   cv='prefit')
cal_knn_1.fit(X_test, y_test)

# Calibration Method 2 - seems like it's the better option
cal_svc_2 = CalibratedClassifierCV(base_estimator=svm.SVC(probability=True, gamma='scale', C=c_value), ensemble=True, cv=cv)
cal_svc_2.fit(final_X, final_y)

# 9 n_neigh best value?
cal_knn_2 = CalibratedClassifierCV(base_estimator=KNeighborsClassifier(p=1, n_neighbors=9), ensemble=True, cv=cv)
cal_knn_2.fit(final_X, final_y)

#pickle.dump(clf_svm, open('trained_models/mod8.bin', 'wb'))



#########

print_cm(cal_svc_1, "Calibrated SVC 1", X_test_diff, y_test_diff)
print_cm(cal_knn_1, "Calibrated KNN 1", X_test_diff, y_test_diff)
print_cm(cal_svc_2, "Calibrated SVC 2", X_test_diff, y_test_diff)
print_cm(cal_knn_2, "Calibrated KNN 2", X_test_diff, y_test_diff)
print_cm(clf_svc, "Uncalibrated SVC", X_test_diff, y_test_diff)
print_cm(clf_knn, "Uncalibrated KNN", X_test_diff, y_test_diff)

clf_list = [(cal_svc_2, 'Calibrated SVC'), (cal_knn_2, 'Calibrated KNN')]
print_learning_curves(clf_list, X_test, y_test, cv)

print_validation_curves(clf_svc, 'Uncalibrated SVC', X_test, y_test, 'C',  param_range=np.linspace(0, 50, 200))
print_validation_curves(clf_knn, 'Uncalibrated KNN', X_test, y_test, 'n_neighbors', np.linspace(0, 50, dtype=int))
print_validation_curves(clf_knn, 'Uncalibrated KNN', X_test, y_test, 'p', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print_validation_curves(cal_svc_2, 'Calibrated SVC', X_test, y_test, 'C',  param_range=np.linspace(0, 50, 200),
                        calibrated=True, X_test_diff=X_test_diff, y_test_diff=y_test_diff)
print_validation_curves(cal_knn_2, 'Calibrated KNN', X_test, y_test, 'p', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        calibrated=True, X_test_diff=X_test_diff, y_test_diff=y_test_diff)
test_scores = print_validation_curves(cal_knn_2, 'Calibrated KNN', X_test, y_test, 'n_neighbors', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        calibrated=True, X_test_diff=X_test_diff, y_test_diff=y_test_diff)




knn_tmp = KNeighborsClassifier(p=1, n_neighbors=9)
knn_tmp_cal = CalibratedClassifierCV(base_estimator=KNeighborsClassifier(p=1, n_neighbors=9), cv=cv)

svc_tmp = svm.SVC(probability=True, gamma='scale', C=c_value)
svc_tmp_cal = CalibratedClassifierCV(base_estimator=svm.SVC(probability=True, gamma='scale', C=c_value), cv=cv)


print_roc_curves(knn_tmp, "Uncalibrated KNN", final_X, final_y, X_test_diff, y_test_diff)
print_roc_curves(knn_tmp_cal, "Calibrated KNN", final_X, final_y, X_test_diff, y_test_diff)

print_roc_curves(svc_tmp, "Uncalibrated SVC", final_X, final_y, X_test_diff, y_test_diff)
print_roc_curves(svc_tmp_cal, "Calibrated SVC", final_X, final_y, X_test_diff, y_test_diff)


svc_tmp = svm.LinearSVC(C=c_value)
svc_tmp_cal = CalibratedClassifierCV(base_estimator=svm.SVC(probability=True, gamma='scale', C=c_value), cv=cv)
svc_tmp_cal.fit(final_X, final_y)
print_cm(svc_tmp_cal, "Linear SVC", X_test_diff, y_test_diff)



final_y_tmp = label_binarize(final_y, classes=[0, 1, 2, 3])
y_test_diff_tmp = label_binarize(y_test_diff, classes=[0,1,2,3])


n_classes = final_y_tmp.shape[1]

clf = OneVsRestClassifier(cal_knn_tmp)
clf.fit(final_X, final_y_tmp)

y_score = clf.predict_proba(X_test_diff)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_diff_tmp[:, i], y_score[:, i])
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
        lw=2,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

np.argmax(test_scores, axis=0)
np.argmax(test_scores, axis=1)
np.mean(test_scores.reshape(-1, 3), axis=1)

counter = 0
mean_row = 0
test_scores_mean = []
for x in test_scores:
    for y in x:
        counter += 1
        mean_row += y

    mean_row /= counter

    test_scores_mean.append(mean_row)
    mean_row = 0
    counter = 0
np.argmax(test_scores_mean, axis=0)




print_calibration_curves_uncal(KNeighborsClassifier(p=1, n_neighbors=9), "KNN", X_train, y_train, X_test, y_test)
print_calibration_curves(KNeighborsClassifier(p=1, n_neighbors=9), "KNN", X_train, y_train, X_test, y_test, X_test_diff, y_test_diff)
print_calibration_curves_2(KNeighborsClassifier(p=1, n_neighbors=9), "KNN", final_X, final_y, cv, X_test_diff, y_test_diff)



temp_clf_svc = svm.SVC(probability=True, gamma='scale', C=c_value)
print_calibration_curves_uncal(temp_clf_svc, "SVC", X_train, y_train, X_test, y_test)
print_calibration_curves(temp_clf_svc, "SVC", X_train, y_train, X_test, y_test, X_test_diff, y_test_diff)
print_calibration_curves_2(temp_clf_svc, "SVC 2", final_X, final_y, cv, X_test_diff, y_test_diff)




###########################################################################################################################





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

