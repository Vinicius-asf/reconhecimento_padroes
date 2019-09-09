# from pandas import read_csv, Series
from numpy import split, cov, mean, transpose
from numpy.random import rand, exponential
from numpy.linalg import det, inv, solve
from math import sqrt, pi
from random import expovariate
import csv

from ex2 import split_for_train_test

tax = 0.2

def pdf_N(x, m, K, n):
  fdp = []
  detK = det(K)
  for i in x:
    A = 1/sqrt(((2*pi)**n)*detK)
    sub = list(i-m)
    aux = -0.5*(transpose(sub)@inv(K)@(sub))
    B = expovariate(aux)
    fdp.append(A*B)
  return fdp

def classifier_N (x, m0, m1, K0, K1, n, p0, p1, indx = 0):
  v0 = pdf_N(x,m0,K0,n)
  v1 = pdf_N(x,m1,K1,n)
  rslt_f = []
  for i in range(len(v0)):
    comp = v0[i]*p0/v1[i]*p1
    if indx == 0:
      if comp >= 1:
        rslt_f.append(1)  
      else :
        rslt_f.append(2)
    else:
      if comp >= 1:
        rslt_f.append(2)  
      else :
        rslt_f.append(1)
  return rslt_f


# 𝑝𝑑𝑓𝑛𝑣𝑎𝑟 < −𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛(𝑥, 𝑚, 𝐾, 𝑛)((1/(𝑠𝑞𝑟𝑡((2*𝑝𝑖)𝑛*(𝑑𝑒𝑡(𝐾)))))*𝑒𝑥𝑝(−0.5*(𝑡(𝑥−𝑚)% * %(𝑠𝑜𝑙𝑣𝑒(𝐾))% * %(𝑥 − 𝑚))))

if __name__ == "__main__":
  heart_1 = []
  heart_2 = []
  with open('C:\\Users\\vinic\\OneDrive\\Faculdade\\Reconhecimento de padrões\\exercicios\\heart.csv', 'r', newline='') as csv_heart:
    csv_reader = csv.reader(csv_heart, delimiter=',')
    for row in csv_reader:
      # print(row[13])
      if row[13] == '1':
        heart_1.append([float(row[i]) for i in range(len(row)-1)])
      elif row[13] == '2':
        heart_2.append([float(row[i]) for i in range(len(row)-1)])
      else:
        print(row)
  
  msk = rand(len(heart_1)) < tax

  train_1, test_1 = split_for_train_test(heart_1,msk)
  train_2, test_2 = split_for_train_test(heart_2,msk)

  priori_1 = len(heart_1)/(len(heart_1)+len(heart_2))
  priori_2 = len(heart_2)/(len(heart_1)+len(heart_2))

  u1 = mean(train_1,axis=0)
  u2 = mean(train_2,axis=0)

  cov1 = cov(train_1, rowvar=False)
  cov2 = cov(train_2, rowvar=False)

  rslt = []
  rslt.append(classifier_N(test_1,u1,u2,cov1,cov2,13,priori_1,priori_2, indx=1))
  rslt.append(classifier_N(test_2,u1,u2,cov1,cov2,13,priori_1,priori_2))

  # print(rslt[0])
  cnt_1 = 0
  cnt_2 = 0
  for i in rslt[0]:
    if i == 1:
      cnt_1+=1
  for i in rslt[1]:
    if i == 2:
      cnt_2+=1
  
  acc1 = cnt_1/len(rslt[0])
  acc2 = cnt_2/len(rslt[1])

  print(acc1)
  print(acc2)