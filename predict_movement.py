import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
walk_y = np.zeros(n_samples)


side_left_X = pickle.load(open('datasets_to_compile/prediction_list_sideLeft.bin', 'rb'))
side_left_X = np.array(side_left_X)
n_samples, nx, ny = side_left_X.shape
side_left_X_r = side_left_X.reshape((n_samples, nx*ny))
side_left_y = np.ones(n_samples)

#side_right_X = pickle.load(open('datasets_to_compile/prediction_list_sideRight.bin', 'rb'))
#side_right_X = np.array(side_right_X)
#n_samples, nx, ny = side_right_X.shape
#side_right_X_r = side_right_X.reshape((n_samples, nx*ny))
#side_right_y = 1

#run_X = pickle.load(open('datasets_to_compile/prediction_list_run1.bin', 'rb'))
#run_X = np.array(run_X)
#n_samples, nx, ny = run_X.shape
#run_X_r = run_X.reshape((n_samples, nx*ny))
#run_y = np.full(n_samples, 2)


final_X = np.vstack((walk_X_r, side_left_X_r))
final_y = np.append(walk_y, side_left_y)


# let's mix it up boy
np.random.seed(0)
indices = np.random.permutation(len(final_X))
X_train = final_X[indices[:-10]]
y_train = final_y[indices[:-10]]


X_test = final_X[indices[-10:]]
y_test = final_y[indices[-10:]]

knn = KNeighborsClassifier(n_neighbors=5)
clf = svm.SVC(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



knn.fit(X_train, y_train)
pickle.dump(knn, open('trained_models/model17.bin', 'wb'))


# when crouching it should count as walking\running
# not finding correctly when going down while walking


# DOCS

# 1) Tra camminata e corsa non c'è vera differenza di movimento, sono molto simili, cambia solo accelerazione


# HISTORY TEMP

# MODEL 14: non identifica correttamente la corsa. Dai test risulta che ancora la vede come rumore o come stato di quiete
# MODEL 15: non ci serve lo stato di quiete se lo possiamo determinare in maniera più easy. Evita possibili sbagli e aumenta chacne di sbagliare
# MODEL 16: altri test, nulla da segnalare
# MODEL 17: eliminata corsa (proviamo a determinarla con media ora) e aggiunto side left