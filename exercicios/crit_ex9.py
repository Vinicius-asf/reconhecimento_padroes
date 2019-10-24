import csv
from numpy.random import rand
from numpy import cov, mean
from operator import indexOf
import matplotlib.pyplot as plt

from ex4 import pdf_N
from ex2 import split_for_train_test
from ex6_kmeans import v_Kmeans

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
  
  spiral = [[*spiral_1[0]+spiral_2[0]],[*spiral_1[1]+spiral_2[1]]]

  # folds = 10
  # msk = rand(len(spiral_1[0])) < 0.9

  # plt.figure(1)
  # plt.scatter(*spiral_1)
  # plt.scatter(*spiral_2)
  # plt.show()
  # Dados carregados até aqui
  
  msk = rand(len(spiral[0])) < 0.9
  spiral_train, spiral_test = split_for_train_test(spiral, msk, 2)

  spiral_t_train, spiral_t_test = split_for_train_test(spiral_t,msk)
  k_done = False
  #Dados separados para treinamento e testes
  for k in range(2,30):
    mean_list = []
    cov_list = []
    priori = []
    pdfs = []
    c_epoch, clusters = v_Kmeans(spiral_train,k)
    # print (clusters)
    c_data = [[]]*k

    for x in range(len(clusters)):
      c_data[clusters[x]].append(spiral_t_train[x])

    # Treinamento feito - hora dos testes
    if k_done:
      for j in range(k):
        cluster_data = [
          [x for x in spiral_train[0] if clusters[indexOf(spiral_train[0],x)] == j],
          [x for x in spiral_train[1] if clusters[indexOf(spiral_train[1],x)] == j]
        ]
        # print(cluster_data)
        mean_list.append(mean(cluster_data,axis=1))
        cov_list.append(cov(cluster_data,rowvar=True))
        priori.append(len(cluster_data)/len(spiral_train))

    # for i in range(len(spiral_test)):
    if k_done:
      pdf_data_1 = [
        [x for x in spiral_test[0] if spiral_t_test[indexOf(spiral_test[0],x)] == 1],
        [x for x in spiral_test[1] if spiral_t_test[indexOf(spiral_test[1],x)] == 1]
      ]
      pdf_data_2 = [
        [x for x in spiral_test[0] if spiral_t_test[indexOf(spiral_test[0],x)] == 2],
        [x for x in spiral_test[1] if spiral_t_test[indexOf(spiral_test[1],x)] == 2]
      ]
      for j in range(k):
        pdfs.append(pdf_N(pdf_data_1, mean_list[k],cov_list[k],2))
        pdfs.append(pdf_N(pdf_data_2, mean_list[k],cov_list[k],2))
        # test_2 = pdf_N(pdf_data, mean_list[k],cov_list[k],2)