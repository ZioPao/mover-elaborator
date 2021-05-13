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


knn.fit(X_train, y_train)

#pickle.dump(knn, open('model3.bin', 'wb'))
#loaded_model = pickle.load(open('model.bin', 'rb'))

test = knn.predict(X_test)

print("trained X_TEST")
print(X_test)

print("trained y_TEST")
print(y_test)