from statistics import mean
import numpy as np
import pickle
import pandas as pd
from numpy.ma import std
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score , train_test_split, permutation_test_score, ShuffleSplit, learning_curve
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
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


#test = np.insert(final_X, int(0), final_y, axis=1)
#pd.DataFrame(test).to_csv("test.csv")


X_train, X_test, y_train, y_test = train_test_split(final_X, final_y, test_size=0.3, random_state=0)
# weights
#weight = np.ones(len(final_y))
#weight[0:len(walk_y)] *= 3
#weight[len(walk_y):len(run_y)] *= 2

# let's mix it up boy
#np.random.seed(0)
#test_sub = 50
#indices = np.random.permutation(len(final_X))
#X_train = final_X[indices[:-test_sub]]
#y_train = final_y[indices[:-test_sub]]
#weight_train = weight[indices[:-test_sub]]

#X_test = final_X[indices[-test_sub:]]
#y_test = final_y[indices[-test_sub:]]

c_value = 12
cv = ShuffleSplit(n_splits=8, random_state=69)
clf_svm = svm.SVC(probability=True, gamma='scale', C=c_value)        # More than 1 will get better result, 0.83 vs 0.93. But is it overfitting?
clf_svm = CalibratedClassifierCV(base_estimator=clf_svm, cv=cv)

clf_knn = KNeighborsClassifier()
clf_knn = CalibratedClassifierCV(base_estimator=clf_knn, cv=cv)








#clf.fit(X_train, y_train, sample_weight=weight_train)
clf.fit(X_train, y_train)

# VALIDATION CURVE!!!!!!
scores = cross_val_score(clf, final_X, final_y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print("Scoring: %.3f" % clf.score(X_test, y_test))

# confusion matrix
y_pred = clf.predict_proba(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm)
#plt.title('SVM - C=' + str(c_value) + ', accuracy=' + str(accuracy_score(y_test, y_pred)))
plt.title('KNN - n_neighbors=2, accuracy=%.3f' % mean(scores))
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#pickle.dump(clf, open('trained_models/mod6.bin', 'wb'))

score_dataset, perm_scores_dataset, pvalue_dataset = permutation_test_score(
    clf, final_X, final_y, scoring="accuracy", cv=cv, n_permutations=1000, n_jobs=-1
)


fig, ax = plt.subplots()

ax.hist(perm_scores_dataset, bins=20, density=True)
ax.axvline(score_dataset, ls="--", color="r")
score_label = f"Score on original\ndata: {score_dataset:.2f}\n(p-value: {pvalue_dataset:.3f})"
ax.text(0.7, 10, score_label, fontsize=12)
ax.set_xlabel("Accuracy score")
_ = ax.set_ylabel("Probability")

fig, ax = plt.subplots()

ax.hist(perm_scores_rand, bins=20, density=True)
ax.set_xlim(0.13)
ax.axvline(score_rand, ls="--", color="r")
score_label = f"Score on original\ndata: {score_rand:.2f}\n(p-value: {pvalue_rand:.3f})"
ax.text(0.14, 7.5, score_label, fontsize=12)
ax.set_xlabel("Accuracy score")
ax.set_ylabel("Probability")
plt.show()


# when crouching it should count as walking\running
# not finding correctly when going down while walking

pickle.dump(all_frames, open('datasets_to_compile/prediction_list_run3.bin', 'wb'))


# DOCS

# 1) Tra camminata e corsa non c'è vera differenza di movimento, sono molto simili, cambia solo accelerazione


# HISTORY TEMP

# MODEL 14: non identifica correttamente la corsa. Dai test risulta che ancora la vede come rumore o come stato di quiete
# MODEL 15: non ci serve lo stato di quiete se lo possiamo determinare in maniera più easy. Evita possibili sbagli e aumenta chacne di sbagliare
# MODEL 16: altri test, nulla da segnalare
# MODEL 17: eliminata corsa (proviamo a determinarla con media ora) e aggiunto side left
# MODEL 18: migliorato 17 semplicemente con side right aggiunto
# MODEL 19: aggiunto salto
# MODEL 20: tornato dopo iato. Ritolto il salto, migliorata precisione dopo implementazione di filtro low pass butterworth.
#           Il salto lo reimplemento una volta che ho capito come differenziarlo maggiormente dalla corsa, che viene confusa troppe volte
#MODEL 21: lost to time
#MODEL 22: passato a CLF piuttosto che KNN, più accurato
# REBOOT
# MOD 4: almost final
# MOD 5: weightning walk and run more than anything else

side_right_X = np.array(tuple_list)
n_samples, nx, ny = side_right_X.shape
side_right_X_r = side_right_X.reshape((n_samples, nx*ny))



counter = 0
counter_y = 0
array_string = []
string_tmp = ''
for x in final_X.flatten():
    counter += 1
    string_tmp += str(x)
    if counter == 306:
        array_string.append([final_y[counter_y], string_tmp])
        string_tmp = ''
        counter = 0

        counter_y += 1
    else:
        string_tmp += ','

np_string = np.array(array_string)
for x in np_string:
    print(x)


pd.DataFrame(np_string).to_csv("test.csv")


test = np.insert(np_string, int(0), final_y, axis=0)

train_sizes, train_scores, valid_scores = learning_curve(
  clf, final_X, final_y, train_sizes=[50, 80, 110], cv=cv)





def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

title = "Learning Curves (SVM)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
plot_learning_curve(
    clf, title, final_X, final_y, axes=axes[:, 0], ylim=(0.5, 1.01), cv=cv, n_jobs=-1)

title = r"Learning Curves (KNN)"
# SVC is more expensive so we do a lower number of CV iterations:
clf = KNeighborsClassifier()
plot_learning_curve(
    clf, title, final_X, final_y, axes=axes[:, 1], ylim=(0.5, 1.01), cv=cv, n_jobs=-1)

plt.show()