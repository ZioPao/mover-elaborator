from statistics import mean
import numpy as np
import pickle
import pandas as pd
from numpy.ma import std
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split, permutation_test_score
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


#test = np.insert(final_X, int(0), final_y, axis=1)
#pd.DataFrame(test).to_csv("test.csv")


X_train, X_test, y_train, y_test = train_test_split(final_X, final_y, test_size=0.3, random_state=0)
# weights
#weight = np.ones(len(final_y))
#weight[0:len(walk_y)] *= 3
#weight[len(walk_y):len(run_y)] *= 2

# let's mix it up boy
#np.random.seed(0)
#test_sub = 50
#indices = np.random.permutation(len(final_X))
#X_train = final_X[indices[:-test_sub]]
#y_train = final_y[indices[:-test_sub]]
#weight_train = weight[indices[:-test_sub]]

#X_test = final_X[indices[-test_sub:]]
#y_test = final_y[indices[-test_sub:]]

c_value = 12
cv = KFold(n_splits=8, random_state=69, shuffle=True)
clf = svm.SVC(probability=True, gamma='scale', C=c_value)        # More than 1 will get better result, 0.83 vs 0.93. But is it overfitting?
clf = CalibratedClassifierCV(base_estimator=clf, cv=cv)


#clf.fit(X_train, y_train, sample_weight=weight_train)
clf.fit(X_train, y_train)

scores = cross_val_score(clf, final_X, final_y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print("Scoring: %.3f" % clf.score(X_test, y_test))

# confusion matrix
y_pred = clf.predict_proba(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm)
#plt.title('SVM - C=' + str(c_value) + ', accuracy=' + str(accuracy_score(y_test, y_pred)))
plt.title('KNN - n_neighbors=2, accuracy=%.3f' % mean(scores))
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#pickle.dump(clf, open('trained_models/mod6.bin', 'wb'))

score_dataset, perm_scores_dataset, pvalue_dataset = permutation_test_score(
    clf, final_X, final_y, scoring="accuracy", cv=cv, n_permutations=1000, n_jobs=-1
)


fig, ax = plt.subplots()

ax.hist(perm_scores_dataset, bins=20, density=True)
ax.axvline(score_dataset, ls="--", color="r")
score_label = f"Score on original\ndata: {score_dataset:.2f}\n(p-value: {pvalue_dataset:.3f})"
ax.text(0.7, 10, score_label, fontsize=12)
ax.set_xlabel("Accuracy score")
_ = ax.set_ylabel("Probability")

fig, ax = plt.subplots()

ax.hist(perm_scores_rand, bins=20, density=True)
ax.set_xlim(0.13)
ax.axvline(score_rand, ls="--", color="r")
score_label = f"Score on original\ndata: {score_rand:.2f}\n(p-value: {pvalue_rand:.3f})"
ax.text(0.14, 7.5, score_label, fontsize=12)
ax.set_xlabel("Accuracy score")
ax.set_ylabel("Probability")
plt.show()


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



counter = 0
counter_y = 0
array_string = []
string_tmp = ''
for x in final_X.flatten():
    counter += 1
    string_tmp += str(x)
    if counter == 306:
        array_string.append([final_y[counter_y], string_tmp])
        string_tmp = ''
        counter = 0

        counter_y += 1
    else:
        string_tmp += ','

np_string = np.array(array_string)
for x in np_string:
    print(x)


pd.DataFrame(np_string).to_csv("test.csv")


test = np.insert(np_string, int(0), final_y, axis=0)

