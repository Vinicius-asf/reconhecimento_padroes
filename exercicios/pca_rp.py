from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv

# fetched = fetch_olivetti_faces()

data = fetch_olivetti_faces().data
data_int = np.rint(data*256).astype(int)
tgt = fetch_olivetti_faces().target

# data = data.T

norm_val = np.mean(data_int,axis=0)
norm_data = data_int - norm_val

cov_norm = np.cov(norm_data,rowvar=False)

eig_val, eig_vec = la.eig(cov_norm)

values = 250

plt.figure(1)
plt.scatter(range(values),eig_val[:values])
plt.ylim((eig_val.min(),eig_val.max()+eig_val.max()/3))
plt.show()

chosen_eig_vec = eig_vec[:values]

new_data = []

for vec in chosen_eig_vec:
  new_data.append(np.dot(norm_data,vec.astype(float)))

new_data = np.array(new_data)
for i in range(10):
  X_train, X_test, y_train, y_test = train_test_split(new_data.T, tgt, test_size=0.5,random_state=109)

  clf = svm.SVC(gamma='scale')
  clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)
  print('Accuracy:',metrics.accuracy_score(y_test,y_pred))

cnf_mtx = metrics.confusion_matrix(y_test,y_pred)

# print(metrics.confusion_matrix(y_test, y_pred))

with open('confusion_result.csv', mode='w',newline='') as conf_res:
    spamwriter = csv.writer(conf_res, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in cnf_mtx:
        spamwriter.writerow(row)

# train_x = X_train[][]
print(1)