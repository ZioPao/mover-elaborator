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
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier
import time

# what if I "fixed" the current dataset with my values?

headers = ['activity', 'x-accel', 'y-accel', 'z-accel']
df_t = pd.read_csv('dataset_test_f.csv', names=headers, low_memory=False)

#df_new_X = pd.DataFrame(added_values, columns=['activity', 'x-accel', 'y-accel', 'z-accel'])
#new_df = pd.concat([df_t, df_new_X])
#new_df.to_csv('new_dataset4.csv')




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

knn = KNeighborsClassifier(n_neighbors=2)
#print("Init CLF")
#clf = svm.SVC()
print("Init Gaussian")
#nc = NearestCentroid()
#rnc = RadiusNeighborsClassifier()

y_train = y_train.astype(float)

print("Training....")
knn.fit(X_train, y_train)
print("Finished training")
pickle.dump(knn, open('trained_models/model7.bin', 'wb'))
#loaded_model = pickle.load(open('model.bin', 'rb'))

test = knn.predict(X_test)

print("trained X_TEST")
print(X_test)

print("trained y_TEST")
print(y_test)



#df_1 = pd.DataFrame(added_values, columns=['activity', 'x-accel', 'y-accel', 'z-accel'])
#df_1.to_csv('dataset_2.csv')


#df_0 = pd.read_csv('dataset_0.csv', names=headers, low_memory=False)
#df_1 = pd.read_csv('dataset_1.csv', names=headers, low_memory=False)
#df_2 = pd.read_csv('dataset_2.csv', names=headers, low_memory=False)

#df_f = df_0.append(df_1.append(df_2))

#df_f.to_csv('dataset_test_f.csv')