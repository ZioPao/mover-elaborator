# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans

headers = ['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel']
df = pd.read_csv('dataset.txt', names=headers)

df_t = df.drop(columns=['user', 'timestamp'])
df_t = df_t.dropna()

# x -> data
# y -> eval

# PRE PROCESSING

# x_train

# fixes z

index = 0
for index, row in df_t.iterrows():
    try:
        df_t.loc[index, 'z-accel'] = row.at['z-accel'][:-1]
    except TypeError:
        pass
    index += 1


X = df_t[['x-accel', 'y-accel', 'z-accel']]  # how do I use other values?
X = X.dropna().to_numpy()

# y_train setup
y = df_t['activity'].array      # needs to be mapped to float values?

index = 0
for var in y:
    if var == 'Jogging':
        y[index] = 0
    if var == 'Walking':
        y[index] = 1
    if var == 'Upstairs':
        y[index] = 2
    if var == 'Downstairs':
        y[index] = 3
    if var == "Standing":
        y[index] = 4

    index += 1

np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[:-10]]
y_train = y[indices[:-10]]
X_test = X[indices[-10:]]

print("CURRENT X_TEST")
print(X_test)
y_test = y[indices[-10:]]

knn = KNeighborsClassifier()

y_train = y_train.astype(float)
knn.fit(X_train, y_train)
test = knn.predict(X_test)

print("trained X_TEST")
print(X_test)

print("trained y_TEST")
print(y_test)





'''


estimators = [('k_means_mov_8', KMeans(n_clusters=8)),
              ('k_means_mov_3', KMeans(n_clusters=3)),
              ('k_means_mov_bad_init', KMeans(n_clusters=3, n_init=1, init='random'))]

fig_num = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fig_num, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(x_train)
    labels = est.labels_

    ax.scatter(x_train[:, 2], x_train[:, 0], x_train[:, 1],
               c=labels.astype(float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fig_num - 1])
    ax.dist = 12
    fig_num = fig_num + 1

# Plot the ground truth
fig = plt.figure(fig_num, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('X', 0),
                    ('Y', 1),
                    ('Z', 2)]:
    ax.text3D(x_train[y_train == label, 2].mean(),
              x_train[y_train == label, 0].mean(),
              x_train[y_train == label, 1].mean() + 1, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y_train, [1, 2, 0]).astype(float)
ax.scatter(x_train[:, x_train], x_train[:, 0], x_train[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()



splitter = ShuffleSplit(n_splits=5, random_state=0, train_size=0.5)
scaler = MinMaxScaler()
clf = GridSearchCV(estimator=svm.SVC(kernel='linear'),
                   param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]})

acc = run(x_train, y_train, splitter, scaler, clf)
print("Hyperparameter estimation (5-fold xval)")
print("    - Best parameters set found on development set:", clf.best_params_)
print("    - Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("        %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

'''#print("Mean test accuracy: {:.1%} +/- {:.1%}\n".format(acc.mean(), 2*acc.std()))
