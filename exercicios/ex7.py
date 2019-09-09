from ex6_kmeans import v_Kmeans
from operator import itemgetter
import matplotlib.pyplot as plt
import csv
from mpl_toolkits import mplot3d
from numpy import zeros, mean, std, arange, meshgrid
from ex1 import pdf2var

if __name__ == "__main__":
  spiral_1 = [[],[]]
  spiral_2 = [[],[]]
  with open('C:\\Users\\vinic\\OneDrive\\Faculdade\\Reconhecimento de padrões\\exercicios\\spirals.csv', 'r', newline='') as csv_spiral:
    csv_reader = csv.reader(csv_spiral, delimiter=',')
    for row in csv_reader:
      # print(row[13])
      if row[2] == '1':
        spiral_1[0].append(float(row[0]))
        spiral_1[1].append(float(row[1]))
      elif row[2] == '2':
        spiral_2[0].append(float(row[0]))
        spiral_2[1].append(float(row[1]))
      else:
        print(row)
  
  lim_inf = -1.10
  lim_ext = 1.10
  discrt = 0.06

  seqi=arange(lim_inf,lim_ext,discrt)
  seqj=arange(lim_inf,lim_ext,discrt)
  m_size = [len(seqi),len(seqj)]

  spiral = [[*spiral_1[0]+spiral_2[0]],[*spiral_1[1]+spiral_2[1]]]
  plt_count = 1
  for k in [5,10,15,20,50]:
    results, cluster = v_Kmeans(spiral,k)
    plt.figure(plt_count)
    plt.title("# de clusters = %i"%(k))
    # list_points = []
    for n in range(k):
      indices = [i for i, x in enumerate(cluster) if x == n]
      points = list(zip(*list(map(lambda x,y: [x,y],*[list(itemgetter(*indices)(spiral[0])),list(itemgetter(*indices)(spiral[1]))]))))
      plt.subplot(1,1,1)
      plt.scatter(*points)
      # list_points.append(points)
    
    # M = [zeros(m_size)*k]

    # u = []
    # s = []
    # for i in range(k):
    #   u.append(mean(list_points[i],axis=1))
    #   s.append(std(list_points[i], axis=1))
    
    # plt.subplot(1,2,1)

    # ci = 0
    # for n in range(k):
    #   for i in seqi:
    #     ci += 1 
    #     cj = 0
    #     for j in seqj:
    #       cj += 1
    #       M[n][ci-1,cj-1] = pdf2var(i,j,*u,*s,0)
      
    plt_count+=1
  plt.show()