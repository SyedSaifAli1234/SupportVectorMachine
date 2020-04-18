import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from statistics import mean

#Retrieving the dataset

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

#KPerforming Fold cross validation with k=5
#shuffling the data

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(iris_X)
print(kf)







#Linear SVM
#trying diffferent values of regularziation parameter c and determining c with best results
# vals_against_acc = {}
# for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
#
#     accuracies = []
#     for train_index, test_index in kf.split(iris_X):
#         #print("TRAIN:", len(train_index), "TEST:", len(test_index))
#         X_train, X_val = iris_X[train_index], iris_X[test_index]
#         y_train, y_val = iris_y[train_index], iris_y[test_index]
#
#         svc = svm.SVC(kernel='linear', C=c)
#         svc.fit(X_train, y_train)
#         predictions = svc.predict(X_val)
#         acc = (predictions == y_val).mean()
#         accuracies.append(acc)
#
#     vals_against_acc[str(c)] = mean(accuracies)
#     print("The Avg. Accuracy is : {} \t for C =  {}".format(round(mean(accuracies), 4), c))
#
# best_C = max(vals_against_acc, key=vals_against_acc.get)
# print('best value of c in linear SVM is:', best_C)

#Non - Linear SVM
#Performing Grid Search on gamma and c, to get best values
vals_against_acc2 = {}

print("(c, gamma) -> Avg. Accuracy")
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    for gamma in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10]:
        accuracies = []
        for train_index, test_index in kf.split(iris_X):
            #print("TRAIN:", len(train_index), "TEST:", len(test_index))
            X_train, X_val = iris_X[train_index], iris_X[test_index]
            y_train, y_val = iris_y[train_index], iris_y[test_index]

            svc = svm.SVC(kernel='linear', C=c)
            svc.fit(X_train, y_train)
            predictions = svc.predict(X_val)
            acc = (predictions == y_val).mean()
            accuracies.append(acc)

        vals_against_acc2[(c, gamma)] = mean(accuracies)
        #print("({}, {}) -> Avg. Accuracy : {} ".format(c, gamma, round(mean(accuracies), 4)))


for key, value in vals_against_acc2.items():
    print(key, " - ",  value)

optimal = max(vals_against_acc2, key=vals_against_acc2.get)
print('best value of (c, gamma) for non-linear SVM are:', optimal)


optimal_C, optimal_gamma = optimal[0], optimal[1]


#Visualizing the results

def plot_contours(ax, clf, xx, yy, **params):
	"""Plot the decision boundaries for a classifier. 
	Parameters 
	---------- 
	ax: matplotlib axes object 
	clf: a classifier 
	xx: meshgrid ndarray yy:
	meshgrid ndarray params:
	dictionary of params to pass to contourf, optional 
	"""
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

svc = svm.SVC(kernel='rbf', gamma=optimal_gamma, C=optimal_C)
pred = svc.fit(iris_X_train, iris_y_train)
U, V = iris_X_train[:, 0], iris_X_train[:, 1]
xx, yy = make_meshgrid(U, V)
figsize = 10
fig = plt.figure(figsize=(figsize,figsize))
ax = plt.subplot(111)
plot_contours(ax, svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(U, V, c=iris_y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())

ax.set_ylim(yy.min(), yy.max())
plt.show()