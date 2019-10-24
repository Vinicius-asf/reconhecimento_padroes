from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv

data,target = datasets.load_breast_cancer(True)

norm_val = np.mean(data)
norm_data = data - norm_val

cov_norm = np.cov(norm_data,rowvar=False)

eig_val, eig_vec = la.eig(cov_norm)

rslt = []


for values in range(2,31):
# values = 5

  # plt.figure(values-1)
  # plt.scatter(range(values),eig_val[:values])
  # plt.ylim((eig_val.min(),eig_val.max()+eig_val.max()/3))
  # plt.title("Quantidade de componentes: %i"%values)
  # plt.savefig('fig%i.png'%values)

  chosen_eig_vec = eig_vec[:values]

  new_data = []

  for vec in chosen_eig_vec:
    new_data.append(np.dot(norm_data,vec.astype(float)))

  new_data = np.array(new_data)

  X_train, X_test, y_train, y_test = train_test_split(new_data.T, target, test_size=0.3,random_state=109)

  clf = svm.SVC(C=10, gamma='scale')

  scores = cross_val_score(clf,new_data.T,target,cv=10)
  print("Features: %i Accuracy: %0.2f (+/- %0.2f)" % (values,scores.mean(), scores.std() * 2))
  
  rslt.append(scores)

with open('pca_svm_rslt.csv',mode='w',newline='') as log_file:
    spamwriter = csv.writer(log_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in rslt:
        spamwriter.writerow(row)

print(1)