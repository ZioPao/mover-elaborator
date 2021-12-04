import numpy as np
import pickle
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

walk_X = pickle.load(open('datasets_to_compile/walk.bin', 'rb'))
walk_X = np.array(walk_X)
n_samples, nx, ny = walk_X.shape
walk_X_r = walk_X.reshape((n_samples, nx*ny))
walk_y = np.zeros(n_samples)

run_X = pickle.load(open('datasets_to_compile/run.bin', 'rb'))
run_X = np.array(run_X)
n_samples, nx, ny = run_X.shape
run_X_r = run_X.reshape((n_samples, nx*ny))
run_y = np.full(n_samples, 1)

side_left_X = pickle.load(open('datasets_to_compile/side_left.bin', 'rb'))
side_left_X = np.array(side_left_X)
n_samples, nx, ny = side_left_X.shape
side_left_X_r = side_left_X.reshape((n_samples, nx*ny))
side_left_y = np.full(n_samples, 2)

side_right_X = pickle.load(open('datasets_to_compile/side_right.bin', 'rb'))
side_right_X = np.array(side_right_X)
n_samples, nx, ny = side_right_X.shape
side_right_X_r = side_right_X.reshape((n_samples, nx*ny))
side_right_y = np.full(n_samples, 3)


final_X = np.vstack((walk_X_r, run_X_r))
final_X = np.vstack((final_X, side_left_X_r))
final_X = np.vstack((final_X, side_right_X_r))


final_y = np.append(walk_y, run_y)
final_y = np.append(final_y, side_left_y)
final_y = np.append(final_y, side_right_y)

# weight
weight = np.ones(len(final_y))
weight[0:len(walk_y)] *= 3
weight[len(walk_y):len(run_y)] *= 2


# let's mix it up boy
np.random.seed(0)
test_sub = 50
indices = np.random.permutation(len(final_X))
X_train = final_X[indices[:-test_sub]]
y_train = final_y[indices[:-test_sub]]
weight_train = weight[indices[:-test_sub]]

X_test = final_X[indices[-test_sub:]]
y_test = final_y[indices[-test_sub:]]

c_value = 12
#clf = svm.SVC(probability=True, gamma='scale', C=c_value)        # More than 1 will get better result, 0.83 vs 0.93. But is it overfitting?
clf = KNeighborsClassifier(n_neighbors=2, weights='distance')
#clf.fit(X_train, y_train, sample_weight=weight_train)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm)
#plt.title('SVM - C=' + str(c_value) + ', accuracy=' + str(accuracy_score(y_test, y_pred)))
plt.title('KNN - n_neighbors=2, accuracy=' + str(accuracy_score(y_test, y_pred)))
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print(accuracy_score(y_test, y_pred))


pickle.dump(clf, open('trained_models/mod5.bin', 'wb'))


# when crouching it should count as walking\running
# not finding correctly when going down while walking

pickle.dump(all_frames, open('datasets_to_compile/prediction_list_run3.bin', 'wb'))


# DOCS

# 1) Tra camminata e corsa non c'è vera differenza di movimento, sono molto simili, cambia solo accelerazione


# HISTORY TEMP

# MODEL 14: non identifica correttamente la corsa. Dai test risulta che ancora la vede come rumore o come stato di quiete
# MODEL 15: non ci serve lo stato di quiete se lo possiamo determinare in maniera più easy. Evita possibili sbagli e aumenta chacne di sbagliare
# MODEL 16: altri test, nulla da segnalare
# MODEL 17: eliminata corsa (proviamo a determinarla con media ora) e aggiunto side left
# MODEL 18: migliorato 17 semplicemente con side right aggiunto
# MODEL 19: aggiunto salto
# MODEL 20: tornato dopo iato. Ritolto il salto, migliorata precisione dopo implementazione di filtro low pass butterworth.
#           Il salto lo reimplemento una volta che ho capito come differenziarlo maggiormente dalla corsa, che viene confusa troppe volte
#MODEL 21: lost to time
#MODEL 22: passato a CLF piuttosto che KNN, più accurato
# REBOOT
# MOD 4: almost final
# MOD 5: weightning walk and run more than anything else

side_right_X = np.array(tuple_list)
n_samples, nx, ny = side_right_X.shape
side_right_X_r = side_right_X.reshape((n_samples, nx*ny))