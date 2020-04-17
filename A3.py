import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from statistics import mean
iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-40]]
iris_y_train = iris_y[indices[:-40]]
iris_X_valid = iris_X[indices[-40:-20]]
iris_y_valid = iris_y[indices[-40:-20]]
iris_X_test = iris_X[indices[-20:]]
iris_y_test = iris_y[indices[-20:]]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf.get_n_splits(iris_X)
print(kf)

params = {}
for c in np.geomspace(1e-3, 1e3, 7).tolist():

    cumulative = []
    for train_index, val_index in kf.split(iris_X):
        #   print("TRAIN:", len(train_index), "TEST:", len(val_index))
        X_train, X_val = iris_X[train_index], iris_X[val_index]
        y_train, y_val = iris_y[train_index], iris_y[val_index]

        svc = svm.SVC(kernel='linear', C=c, random_state=42)
        svc.fit(X_train, y_train)
        predictions = svc.predict(X_val)
        acc = (predictions == y_val).mean()
        cumulative.append(acc)

    params[str(c)] = mean(cumulative)
    print("Value of C: {} \t Avg. Accuracy: {:1f}".format(c, round(mean(cumulative), 4)))

lists = sorted(params.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.figure(figsize=(8,6))
plt.xlabel('Values of C')
plt.ylabel('Avg. Accuracy')
plt.plot(x, y)
plt.show()