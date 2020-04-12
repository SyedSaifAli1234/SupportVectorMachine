import sklearn

from sklearn import datasets
iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target