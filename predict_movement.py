import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier


# setup for single types of act
#stop_X = pickle.load(open('datasets_to_compile/prediction_list_stop5.bin', 'rb'))
#stop_X = np.array(stop_X)
#n_samples, nx, ny = stop_X.shape
#stop_X_r = stop_X.reshape((n_samples, nx*ny))
#stop_y = np.zeros(n_samples)

walk_X = pickle.load(open('datasets_to_compile/prediction_list_walk4.bin', 'rb'))
walk_X = np.array(walk_X)
n_samples, nx, ny = walk_X.shape
walk_X_r = walk_X.reshape((n_samples, nx*ny))
walk_y = np.ones(n_samples)


side_left_X = pickle.load(open('datasets_to_compile/prediction_list_sideLeft.bin', 'rb'))
side_left_X = np.array(side_left_X)
n_samples, nx, ny = side_left_X.shape
side_left_X_r = side_left_X.reshape((n_samples, nx*ny))
side_left_y = 1

side_right_X = pickle.load(open('datasets_to_compile/prediction_list_sideRight.bin', 'rb'))
side_right_X = np.array(side_right_X)
n_samples, nx, ny = side_right_X.shape
side_right_X_r = side_right_X.reshape((n_samples, nx*ny))
side_right_y = 1

run_X = pickle.load(open('datasets_to_compile/prediction_list_run1.bin', 'rb'))
run_X = np.array(run_X)
n_samples, nx, ny = run_X.shape
run_X_r = run_X.reshape((n_samples, nx*ny))
run_y = np.full(n_samples, 2)

final_X = np.vstack((walk_X_r, run_X_r))
final_y = np.append(walk_y, run_y)


# let's mix it up boy
np.random.seed(0)
indices = np.random.permutation(len(final_X))
X_train = final_X[indices[:-10]]
y_train = final_y[indices[:-10]]


X_test = final_X[indices[-10:]]
y_test = final_y[indices[-10:]]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pickle.dump(knn, open('trained_models/model16.bin', 'wb'))

# test
test = knn.predict(X_test)

# when crouching it should count as walking\running
# not finding correctly when going down while walking


# DOCS

# 1) Tra camminata e corsa non c'è vera differenza di movimento, sono molto simili, cambia solo accelerazione


# HISTORY TEMP

# MODEL 14: non identifica correttamente la corsa. Dai test risulta che ancora la vede come rumore o come stato di quiete
# MODEL 15: non ci serve lo stato di quiete se lo possiamo determinare in maniera più easy. Evita possibili sbagli e aumenta chacne di sbagliare
# MODEL 16: altri test, nulla da segnalare
# MODEL