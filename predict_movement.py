import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import filterpy
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier
import time


# 1) Un movimento deve essere dato da più movimenti... 1 secondo di movimento nel training. Da inserire un delta time nella lista passata da arduino



# what if I "fixed" the current dataset with my values?
headers = ['activity', 'mov-list']

print("Setting up X")
X = df_t[]

print("Setting up y")



n_samples, nx, ny = prediction_list_np.shape
X_reshaped = prediction_list_np.reshape((n_samples,nx*ny))

y = np.ones(n_samples)

# dopo
np.random.seed(0)
indices = np.random.permutation(len(X_reshaped))
X_train = X_reshaped[indices[:-10]]
y_train = y[indices[:-10]]
y_train = y_train.astype(float)
X_test = X_reshaped[indices[-10:]]
y_test = y[indices[-10:]]


knn = KNeighborsClassifier(n_neighbors=5)
print("Training....")
knn.fit(X_train, y_train)
print("Finished training")
pickle.dump(knn, open('trained_models/model9.bin', 'wb'))
#loaded_model = pickle.load(open('model.bin', 'rb'))

test = knn.predict(X_test)

print("trained X_TEST")
print(X_test)

print("trained y_TEST")
print(y_test)







#####################################################################

# setup for single types of act
stop_X = pickle.load(open('datasets_to_compile/prediction_list_stop2.bin', 'rb'))
stop_X = np.array(stop_X)
n_samples, nx, ny = stop_X.shape
stop_X_r = stop_X.reshape((n_samples, nx*ny))
stop_y = np.zeros(n_samples)

walk_X = pickle.load(open('datasets_to_compile/prediction_list_walk3.bin', 'rb'))
walk_X = np.array(walk_X)
n_samples, nx, ny = walk_X.shape
walk_X_r = walk_X.reshape((n_samples, nx*ny))
walk_y = np.ones(n_samples)

run_X = #
...

final_X = np.vstack((stop_X_r, walk_X_r))
final_y = np.append(stop_y, walk_y)



# let's mix it up boy
np.random.seed(0)
indices = np.random.permutation(len(final_X))
X_train = final_X[indices[:-10]]
y_train = final_y[indices[:-10]]


X_test = final_X[indices[-10:]]
y_test = final_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pickle.dump(knn, open('trained_models/model11.bin', 'wb'))

# test
test = knn.predict(X_test)

