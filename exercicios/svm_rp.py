from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import csv
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

if __name__ == "__main__":
  # preparação dos dados
  spiral = []
  spiral__t = []
  spiral__1 = []
  spiral__1__t = []
  spiral__2 = []
  spiral__2__t = []
  with open('C:\\Users\\vinic\\OneDrive\\Faculdade\\Reconhecimento de padrões\\exercicios\\spirals.csv', 'r', newline='') as csv_spiral:
    csv_reader = csv.reader(csv_spiral, delimiter=',')
    for row in csv_reader:
      if row[2] == '1':
        spiral__1.append([float(row[0]),float(row[1])])
        spiral__1__t.append(float(row[2]))
      elif row[2] == '2':
        spiral__2.append([float(row[0]),float(row[1])])
        spiral__2__t.append(float(row[2]))
      else:
        print(row)
        continue
      spiral.append([float(row[0]),float(row[1])])
      spiral__t.append(float(row[2]))
  spiral__1 = np.array(spiral__1)
  spiral__2 = np.array(spiral__2)
  spiral__2__t = np.array(spiral__2__t)
  spiral__1__t = np.array(spiral__1__t)
  spiral = np.array(spiral)
  spiral__t = np.array(spiral__t)
  print(spiral[0])
  # separação dos dados para treinamento e teste
  X_train, X_test, y_train, y_test = train_test_split(spiral, spiral__t, test_size=0.2,random_state=109) # 80% training and 30% test

  # setup do svm
  # clf = svm.SVC(gamma='scale')

  # treinamento
  # clf.fit(X_train,y_train)

  # teste
  # y_pred = clf.predict(X_test)

  for kernel in ('linear', 'rbf', 'poly'):
    clf = svm.SVC(kernel=kernel, gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plt.figure()
    plt.clf()
    plt.scatter(spiral[:, 0], spiral[:, 1], c=spiral__t, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = spiral[:, 0].min()
    x_max = spiral[:, 0].max()
    y_min = spiral[:, 1].min()
    y_max = spiral[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)

    print('Accuracy:',metrics.accuracy_score(y_test,y_pred))
  plt.show()
