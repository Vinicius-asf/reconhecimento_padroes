from ex6_kmeans import v_Kmeans
from operator import itemgetter
from math import floor
from numpy.random import rand
from ex2 import split_for_train_test
from ex4 import pdf_N, classifier_N
from random import sample
from itertools import compress
from numpy import mean, cov, array
import matplotlib.pyplot as plt
import csv

def cross_val_group(data, msk, folds = 10):
  train, test = split_for_train_test(data,msk,2)
  fold_size = 0
  if len(train) > 1:
    fold_size = floor(len(train[0])/folds)
  else:
    fold_size = floor(len(train)/folds)
  
  # msk = rand(fold_size) < acc
  pool = range(folds)

  list_folds = []

  for i in range(fold_size):
    l_fold = sample(list(pool),k=len(pool))
    for j in range(len(l_fold)):
      list_folds.append(l_fold[j])

  return train, test, list_folds

if __name__ == "__main__":
  spiral_1 = [[],[]]
  spiral_2 = [[],[]]
  spiral_t = []
  with open('C:\\Users\\vinic\\OneDrive\\Faculdade\\Reconhecimento de padrões\\exercicios\\spirals.csv', 'r', newline='') as csv_spiral:
    csv_reader = csv.reader(csv_spiral, delimiter=',')
    for row in csv_reader:
      # print(row[13])
      if row[2] == '1':
        spiral_1[0].append(float(row[0]))
        spiral_1[1].append(float(row[1]))
        spiral_t.append(row[2])
      elif row[2] == '2':
        spiral_2[0].append(float(row[0]))
        spiral_2[1].append(float(row[1]))
        spiral_t.append(row[2])
      else:
        print(row)
  folds = 10
  msk = rand(len(spiral_1[0])) < 0.9

  sp1_train, sp1_test, sp_list = cross_val_group(spiral_1,msk)
  sp2_train, sp2_test = split_for_train_test(spiral_2,msk,2)
  spt_train, spt_test = split_for_train_test(spiral_t,msk)

  # print(spt_test)
  cross_rslts = []
  folds_dict = {}


  for i in range(folds):
    rslts = []
    train_test_fold = array([x != i for x in sp_list])
    train_fold_1 = [list(compress(sp1_train[0],train_test_fold)),list(compress(sp1_train[1],train_test_fold))]
    test_fold_1 = [list(compress(sp1_train[0],~train_test_fold)),list(compress(sp1_train[1],~train_test_fold))]
    train_fold_2 = [list(compress(sp2_train[0],train_test_fold)),list(compress(sp2_train[1],train_test_fold))]
    test_fold_2 = [list(compress(sp2_train[0],~train_test_fold)),list(compress(sp2_train[1],~train_test_fold))]

    u1 = mean(train_fold_1, axis=1)
    u2 = mean(train_fold_2, axis=1)

    cov1 = cov(train_fold_1,rowvar=True)
    cov2 = cov(train_fold_2,rowvar=True)
    
    priori_1 = len(train_fold_1)/(len(train_fold_1)+len(train_fold_2))
    priori_2 = len(train_fold_2)/(len(train_fold_1)+len(train_fold_2))

    tf1 = list(zip(*test_fold_1))
    tf2 = list(zip(*test_fold_2))


    rslts.append(classifier_N(tf1,u1,u2,cov1,cov2,2,priori_1,priori_2, indx=1))
    rslts.append(classifier_N(tf2,u1,u2,cov1,cov2,2,priori_1,priori_2))

    folds_dict[i] = {
        'mean1' : u1,
        'mean2' : u2,
        'covariance1' : cov1,
        'covariance2' : cov2,
        'priori1' : priori_1,
        'priori2' : priori_2
      }
    
    cnt_1 = 0
    cnt_2 = 0
    for i in rslts[0]:
      if i == 1:
        cnt_1+=1
    for i in rslts[1]:
      if i == 2:
        cnt_2+=1
    
    acc1 = cnt_1/len(rslts[0])
    acc2 = cnt_2/len(rslts[1])

    cross_rslts.append((acc1+acc2)/2)

  print(cross_rslts)


  for n in range(k):
    indices = [i for i, x in enumerate(cluster) if x == n]
    points = list(zip(*list(map(lambda x,y: [x,y],*[list(itemgetter(*indices)(spiral[0])),list(itemgetter(*indices)(spiral[1]))]))))
    plt.subplot(1,1,1)
    plt.scatter(*points)
