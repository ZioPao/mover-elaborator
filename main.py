import numpy as np

from sklearn import svm
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import tensorflow as tf


def plot_dataset(x, y, feat0=0, feat1=1):
    colors = ['b.', 'r.', 'g.', 'k.', 'c.', 'm.']
    class_labels = np.unique(y).astype(int)
    for k in class_labels:
        plt.plot(x[y == k, feat0], x[y == k, feat1], colors[k % 7])


def plot_decision_regions(x, y, classifier, resolution=1e-3):
    # setup marker generator and color map
    colors = ('blue', 'red', 'lightgreen', 'black', 'cyan', 'magenta')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 0.02, x[:, 0].max() + 0.02
    x2_min, x2_max = x[:, 1].min() - 0.02, x[:, 1].max() + 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


def load_mnist(n_samples, tr=False):
    if tr:
        # load mnist training (60,000 samples)
        (x, y), _ = tf.keras.datasets.mnist.load_data()
    else:
        # load mnist test (10,000 samples)
        _, (x, y) = tf.keras.datasets.mnist.load_data()
    # print(x.shape, y.shape)
    # let's reshape and subsample to the first 500 digits
    x = x.reshape(x.shape[0], 784)  # flatten each digit as a row
    x = x[:n_samples, :]  # take the first n_samples digits
    y = y[:n_samples]
    return x, y


def plot_digits(x, y, n=10):
    for i in range(n**2):
        plt.subplot(n,n,i+1)
        plt.imshow(x[i].reshape(28,28), cmap='Greys')
        plt.axis('off')


def run(x, y, splitter, scaler, clf):
    """Take input data (x,y), split it (n times), scale it,
    learn classifier on training data, and evaluate the mean test error.
    """
    acc = np.zeros(shape=(splitter.get_n_splits(),))

    for i, (tr_idx, ts_idx) in enumerate(splitter.split(x, y)):
        xtr = x[tr_idx, :]
        ytr = y[tr_idx]
        xts = x[ts_idx, :]
        yts = y[ts_idx]

        xtr = scaler.fit_transform(xtr)
        xts = scaler.transform(xts)

        clf.fit(xtr, ytr)           #guess what parameter on the param grid is the best
        ypred = clf.predict(xts)
        acc[i] = (ypred == yts).mean()
    return acc


######################################################################################
x, y = load_mnist(n_samples=500)
plt.figure(figsize=(8,8))
plot_digits(x, y)
plt.show()

splitter = ShuffleSplit(n_splits=5, random_state=0, train_size=0.5)
scaler = MinMaxScaler()
clf = GridSearchCV(estimator=svm.SVC(kernel='linear'),
                   param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]})

acc = run(x, y, splitter, scaler, clf)
print("Hyperparameter estimation (5-fold xval)")
print("    - Best parameters set found on development set:", clf.best_params_)
print("    - Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("        %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print("Mean test accuracy: {:.1%} +/- {:.1%}\n".format(acc.mean(), 2*acc.std()))