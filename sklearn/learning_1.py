from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

# print(iris_x[:2, :])
# print(iris_y)

X_train, X_test, Y_train, Y_test = train_test_split(iris_x, iris_y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

print('预测值:\n', knn.predict(X_test))
print('真实值:\n', Y_test)