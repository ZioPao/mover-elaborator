# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans

import time

# what if I "fixed" the current dataset with my values?

headers = ['activity', 'x-accel', 'y-accel', 'z-accel']
df_t = pd.read_csv('new_dataset.csv', names=headers, low_memory=False)

#df_new_X = pd.DataFrame(new_X, columns=['activity', 'x-accel', 'y-accel', 'z-accel'])
#new_df = pd.concat([df_t, df_new_X])
#new_df.to_csv('new_dataset.csv')





# x -> data
# y -> eval

# PRE PROCESSING

# x_train


# Corrected dataset from notepad++ removing ; on every line

print("Setting up X")
X = df_t[['x-accel', 'y-accel', 'z-accel']].to_numpy() # how do I use other values?


print("Setting up y")
y = df_t['activity'].array      # needs to be mapped to float values?


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


# add our values to the list
headers_mod = ['x-accel', 'y-accel', 'z-accel']
stopped_movement = pd.DataFrame(list_to_predict, columns=headers_mod).to_numpy()
indices = np.random.permutation(len(stopped_movement))

stopped_movement = stopped_movement.astype('float16')
stopped_movement = stopped_movement / 1000      #reduction to "normalize" using the original dataset as a reference

stopped_movement = stopped_movement[indices[:-10]]
y_mod = np.array([6] * len(stopped_movement))
y_mod = y_mod[indices[:-10]]



second_test_X = stopped_movement[indices[-10:]]
second_test_y = y_mod[indices[-10:]]

X_train_mod = np.concatenate((X_train, stopped_movement))
y_train = np.append(y_train, y_mod)




knn.fit(X_train_mod, y_train)

pickle.dump(knn, open('model.bin', 'wb'))
#loaded_model = pickle.load(open('model.bin', 'rb'))

test = knn.predict(X_test)

print("trained X_TEST")
print(X_test)

print("trained y_TEST")
print(y_test)

# 4 rows x 2, 8 rows total per second
test_mover = [[28, -24, 172], [96, 104, -84], [12, -88, -36], [112, 48, 52],
              [12, -96, -92], [68, -36, -164], [8, -128, -108], [56, -64, -48],
              [12, -40, -80], [40, -48, 76], [44, -104, -56], [120, -40, -44],
              [-16, -76, -192], [0, -148, -28], [64, -216, 0], [68, -140, -12], [68, -108, -44],
              [-40, -96, -108], [0, -132, -112], [-68, -140, -4], [-1448, 8904, -22020], [7528, -600, 12204],
              [5064, -88, 12544], [-1864, -8020, -30784], [5736, -824, 16383], [3516, -4456, 4972], [-7468, -5092, -19828], [16896, 1396, 16383], [-6472, 13172, 15316], [-12820, 2704, -9644], [20040, 1344, 16383], [-1876, -4500, -18508], [-28676, 14668, 16383], [8476, 5376, 16383], [-544, -11256, 16384], [26016, 6040, 16383], [-32384, -19700, 21372], [-2580, 24632, 16383], [3688, -2024, 6628], [-15664, -7028, 11896], [5524, -7972, 16383], [-9008, -16180, -23768], [2868, -5380, 16383], [-3576, -12448, 22232], [1452, -4968, -84], [-1832, 5052, 3752], [-10548, -18332, 28264], [7452, 29168, 16383], [-2436, 18120, -6772], [2676, 9628, 9556], [408, 1660, -3412], [-7980, 6568, 16383], [-2044, 19148, 16383], [-25016, 10708, 16384], [-5060, -21124, 16383], [-9280, -4372, 16383], [-11216, 14940, 16384], [11844, 6012, 7920], [-7216, 6160, 16383], [-11760, 14220, 16384], [6560, 3856, 16383], [-176, 1736, 16383], [-712, 7924, 16384], [-8084, -32768, 16383], [-16328, -30040, 72], [2256, -2916, -3328], [6756, 11560, 8924], [-32768, 32308, 16384], [-3720, 4356, 16383], [-19856, -18664, 16384], [-2328, -14048, 16383], [-3208, 3520, 16383], [-4136, 16192, -30436], [9396, 11940, 16383], [-1740, 8892, -21280], [3436, 7432, 8000], [5148, 2772, 4176], [4464, 588, 2016], [1432, 1308, -4616], [-1744, -3512, -6716], [1084, -3388, 432], [6360, -4308, 16383], [1840, 408, -2352], [-1328, -324, -17016], [-1044, -612, -4276], [2756, -2980, 2080], [4428, -3800, 8576], [940, -1568, 740], [204, -3056, 392], [-440, -2692, -9604], [3604, -2572, -212], [2744, -1216, 2244], [344, -1076, 744],
              [-400, -2272, -652], [708, 1452, -3592], [3184, 0, 3996], [1004, 1080, -4172], [-324, -264, -1572], [-1004, -1580, -5644], [2072, -2648, 2144]]

test_mover_numpy = np.array(list_to_predict)
test_mover_numpy = test_mover_numpy.astype('float16')
test_mover_numpy = test_mover_numpy / 1000      #reduction to "normalize" using the original dataset as a reference


time1 = time.time()
test2 = knn.predict(test_mover_numpy)
time2 = time.time()

print(time2 - time1)

sublist_x = list()
sublist_y = list()
sublist_z = list()
for value in test_mover_numpy:
    print(value)
    sublist_x.append(value[0]/20)
    sublist_y.append(value[1]/20)
    sublist_z.append(value[2]/20)

counter = range(0, len(sublist_x))
ax_x = plt.axes()
ax_x.scatter(counter, sublist_x, c='b' )

ax_y = plt.axes()
ax_y.scatter(counter, sublist_y, c='r')

ax_z = plt.axes()
ax_z.scatter(counter, sublist_z, c='g')
plt.show()


# type 6

type_6 = 'Stopped'
type_6_int = 6





























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




for x in df_t.iterrows():
    if x[0] == 0:
        print(x)

'''#print("Mean test accuracy: {:.1%} +/- {:.1%}\n".format(acc.mean(), 2*acc.std()))


