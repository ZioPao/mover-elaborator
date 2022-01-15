from itertools import cycle
from matplotlib.font_manager import FontProperties
import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import permutation_test_score, validation_curve, learning_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, label_binarize


def print_cm(clf, title, X_test, y_test):

    # CONFUSION MATRIX SVC
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    cm_disp.plot()
    #plt.title('SVC - C= {c}, accuracy={acc:.3f}'.format(c=c_value, acc=accuracy_score(y_test, y_pred)))

    plt.title('{title} - accuracy={acc:.3f}'.format(title=title, acc=accuracy_score(y_test, y_pred)))
    plt.show()


def print_permutation_plots(clf, cv, X_test, y_test):
    score_dataset, perm_scores_dataset, pvalue_dataset = permutation_test_score(
        clf, X_test, y_test, scoring="accuracy", cv=cv, n_permutations=1000, n_jobs=-1)

    fig, ax = plt.subplots()
    ax.hist(perm_scores_dataset, bins=20, density=True)
    ax.axvline(score_dataset, ls="--", color="r")
    score_label = f"Score on original\ndata: {score_dataset:.2f}\n(p-value: {pvalue_dataset:.3f})"
    ax.text(0.7, 10, score_label, fontsize=12)
    ax.set_xlabel("Accuracy score")
    _ = ax.set_ylabel("Probability")


def print_validation_curves(clf, title, X_test, y_test, param, param_range, calibrated=False,
                            X_test_diff=None, y_test_diff=None):
    if calibrated:
        train_scores, test_scores = validation_curve(clf, X_test_diff, y_test_diff,
                                                     param_name="base_estimator__{param}".format(param=param),
                                                     param_range=param_range, scoring="accuracy", n_jobs=-1)
    else:

        train_scores, test_scores = validation_curve(clf, X_test, y_test,
                                                     param_name="{param}".format(param=param), param_range=param_range,
                                                     scoring="accuracy", n_jobs=-1)
    #base_estimator__ for calibrated but it doeesnt work anymore after fitting
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    #plt.title("Validation Curves for KNN model")
    plt.figure()
    plt.title(title)

    plt.xlabel(param)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw, )
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw, )
    plt.legend(loc="best")

    plt.xscale("linear")
    plt.yscale("linear")
    plt.show()
    return test_scores


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5),):
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


def print_learning_curves(clf_list, X_test, y_test, cv):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))


    plot_learning_curve(clf_list[0][0], clf_list[0][1], X_test, y_test, axes=axes[:, 0], ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
    plot_learning_curve(clf_list[1][0], clf_list[1][1], X_test, y_test, axes=axes[:, 1], ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
    plt.show()



def print_calibration_curves_uncal(clf, title, X_train, y_train, X_test, y_test):

    # Binarize the output
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # Train a model with tfidf-vectorizer and LinearSVC
    #tfidf = TfidfVectorizer()
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    # Fit the model
    #pipe = Pipeline([('tfidf', tfidf), ('clf', clf)])

    # Plot the Calibration Curve for every class
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    targets = range(len(lb.classes_))
    for target in targets:
        prob_pos = clf.predict_proba(X_test)[:, target]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test[:, target], prob_pos, n_bins=10)
        name = lb.classes_[target]

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("The proportion of samples whose class is the positive class")
    ax1.set_xlabel("The mean predicted probability in each bin")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots - {t}'.format(t=title))

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

def print_calibration_curves(clf, title, X_train, y_train, X_test, y_test, X_test_diff, y_test_diff):

    # Binarize the output
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    y_test_diff = lb.transform(y_test_diff)


    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)
    clf = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
    clf.fit(X_test, y_test)


    # Fit the model
    #pipe = Pipeline([('tfidf', tfidf), ('clf', clf)])

    # Plot the Calibration Curve for every class
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    targets = range(len(lb.classes_))
    for target in targets:
        prob_pos = clf.predict_proba(X_test_diff)[:, target]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_diff[:, target], prob_pos, n_bins=10)
        name = lb.classes_[target]

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("The proportion of samples whose class is the positive class")
    ax1.set_xlabel("The mean predicted probability in each bin")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots - Calibrated {t}'.format(t=title))

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

def print_calibration_curves_2(clf, title,final_X, final_y,cv, X_test_diff, y_test_diff):

    # Binarize the output
    lb = LabelBinarizer()
    y_train = lb.fit_transform(final_y)
    y_test_diff = lb.transform(y_test_diff)


    clf = OneVsRestClassifier(clf)
    clf = CalibratedClassifierCV(base_estimator=clf, cv=cv)
    clf.fit(final_X, final_y)

    # Fit the model
    # pipe = Pipeline([('tfidf', tfidf), ('clf', clf)])

    # Plot the Calibration Curve for every class
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    targets = range(len(lb.classes_))
    for target in targets:
        prob_pos = clf.predict_proba(X_test_diff)[:, target]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_diff[:, target], prob_pos, n_bins=10)
        name = lb.classes_[target]

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("The proportion of samples whose class is the positive class")
    ax1.set_xlabel("The mean predicted probability in each bin")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots - Calibrated {t}'.format(t=title))

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

def print_roc_curves(clf, title, final_X, final_y, X_test, y_test):
    # Compute ROC curve and ROC area for each class

    final_y = label_binarize(final_y, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

    n_classes = final_y.shape[1]

    clf = OneVsRestClassifier(clf)
    clf.fit(final_X, final_y)

    y_score = clf.predict_proba(X_test)
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

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "forestgreen"])
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
    plt.title(title)
    plt.legend(loc="lower right")

    #plt.gca().set_position((.1, .3, .8, .6))
    micro_roc_auc_ovo = roc_auc_score(y_test, y_score, multi_class="ovr", average="micro")
    macro_roc_auc_ovo = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")


    font_big = FontProperties()
    font_big.set_size('x-large')
    font_big.set_weight('bold')

    font_medium = FontProperties()
    font_medium.set_size('large')

    #plt.figtext(.1, .15, "ROC AUC Scores - One Vs Rest", fontproperties=font_big)
    #plt.figtext(.1, .1, "Micro: ", fontproperties=font_medium)
    #plt.figtext(.25, .1, micro_roc_auc_ovo)
    #plt.figtext(.1, .07, "Macro: ", fontproperties=font_medium)
    #plt.figtext(.25, .07, macro_roc_auc_ovo)
    #plt.figtext(.1, .04, "Weighted: ", fontproperties=font_medium)
    #plt.figtext(.25, .04, weighted_roc_auc_ovo)


    #plt.figtext(.1, .01, "ROC AUC Scores - One Vs Rest\nMicro: {micro}\nMacro: {macro}\nWeighted: {weighted}".
    #            format(micro=micro_roc_auc_ovo, macro=macro_roc_auc_ovo, weighted=weighted_roc_auc_ovo))

    plt.show()

def print_aoc(clf, final_X, final_y, X_test, y_test):
    final_y = label_binarize(final_y, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

    clf = OneVsRestClassifier(clf)
    clf.fit(final_X, final_y)
    y_prob = clf.predict_proba(X_test)

    print("")
    micro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovr", average="micro")
    print("Micro: {x}".format(x=micro_roc_auc_ovo))
    macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    print("Macro: {x}".format(x=macro_roc_auc_ovo))
    weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print("Weighted: {x}".format(x=weighted_roc_auc_ovo))
