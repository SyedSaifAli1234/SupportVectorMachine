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
#plt.show()

parameters = {'C':np.geomspace(1e-3, 1e3, 7).tolist(), 'gamma': np.geomspace(1e-2, 1e1, 7).tolist()}
svc = svm.SVC(kernel='rbf', random_state=42)

#For the parameter 'cv' with integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used
non_linear_clf = GridSearchCV(svc, parameters, refit=True, cv=5, n_jobs=-1, scoring='accuracy')
non_linear_clf.fit(iris_X, iris_y)

optimised_nonlinear_SVM = non_linear_clf.best_estimator_
print(optimised_nonlinear_SVM)
print(non_linear_clf.best_score_)
print(non_linear_clf.best_params_)
best_C = non_linear_clf.best_params_['C']
best_gamma = non_linear_clf.best_params_['gamma']

def plot_contours(ax, clf, xx, yy, **params):
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  out = ax.contourf(xx, yy, Z, **params)
  return out
def make_meshgrid(x, y, h=.02):
  x_min, x_max = x.min() - 1, x.max() + 1
  y_min, y_max = y.min() - 1, y.max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
  np.arange(y_min, y_max, h))
  return xx, yy

svc = svm.SVC(kernel='rbf', gamma=best_gamma, C=best_C, random_state=42)
pred = svc.fit(iris_X_train, iris_y_train)
U, V = iris_X_train[:, 0], iris_X_train[:, 1]
xx, yy = make_meshgrid(U, V)
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)
plot_contours(ax, svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(U, V, c=iris_y_train, cmap=plt.cm.cool, s=30, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())

ax.set_ylim(yy.min(), yy.max())
plt.show()