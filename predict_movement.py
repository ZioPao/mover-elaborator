mport numpy as np
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
