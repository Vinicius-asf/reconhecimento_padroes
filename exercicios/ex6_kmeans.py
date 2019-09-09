from random import sample
from math import sqrt
from operator import itemgetter
from scipy.spatial.distance import euclidean
from itertools import compress, repeat
from numpy.random import normal
import matplotlib.pyplot as plt

def create_4norm(std):
  points = 100
  arr_x_1 = list(normal(2,std,points))
  arr_y_1 = list(normal(2,std,points))

  arr_x_2 = list(normal(4,std,points))
  arr_y_2 = list(normal(4,std,points))

  arr_x_3 = list(normal(2,std,points))
  arr_y_3 = list(normal(4,std,points))

  arr_x_4 = list(normal(4,std,points))
  arr_y_4 = list(normal(2,std,points))
  return [arr_x_1+arr_x_2+arr_x_3+arr_x_4,arr_y_1+arr_y_2+arr_y_3+arr_y_4]

def v_Kmeans(arr, n_clusters = 2, iter = 100):
  centers = sample(range(len(arr[0])),k=n_clusters)
  # print(centers)
  centers = list(map(lambda x,y: [x,y],*[list(itemgetter(*centers)(arr[0])),list(itemgetter(*centers)(arr[1]))]))
  # print(centers)
  i_count = 0
  centers_epoch = []
  while i_count < iter:
    clusters = [0]*len(arr[0]) # lista de pertencimento de um ponto a um cluster
    distance = [[] for i in range(n_clusters)] # lista de distancia de um ponto até um centro
    i_count+=1
    # print('Iteracao %i'%i_count)
    # print(distance,i_count)
    # calculando a distancia dos pontos para os centros
    for i in range(n_clusters):
      for n in range(len(arr[0])):
        e_distance = euclidean([arr[0][n],arr[1][n]],[centers[i][0],centers[i][1]])
        # e_distance = distance_euclidian(arr[0][n],arr[1][n],centers[i][0],centers[i][1])
        distance[i].append(e_distance)
      # clusters.append(dist.index(min(dist)))
    
    # designando pontos a um cluster
    for n in range(len(arr[0])):
      dist = []
      for i in range(n_clusters):
        dist.append(distance[i][n])
      # for i in range(n_clusters):
      clusters[n] = dist.index(min(dist))
    # print(clusters)
    # Recalcular os centros
    sum_s = [[0,0] for i in range(n_clusters)]
    count_s = [0]*n_clusters
    for n in range(len(arr[0])):
      p_cluster = clusters[n]
      count_s[p_cluster]+=1
      # for i in range(n_clusters):
      sum_s[p_cluster][0]+=arr[0][n]
      sum_s[p_cluster][1]+=arr[1][n]
    # print(sum_s,len(sum_s[0]),len(sum_s[1]))
    # print(count_s)
    for i in range(n_clusters):
      centers[i]=[sum_s[i][0]/count_s[i],sum_s[i][1]/count_s[i]]
    # print(centers)
    centers_epoch.append(centers)
    if len(centers_epoch)>iter/4 and centers_epoch[i_count-1]==centers_epoch[i_count-2]:
      break
  return centers_epoch, clusters

if __name__ == "__main__":
  plt_count = 1
  for std in [x*0.1 for x in range(3,9,2)]:
    # axs[0].plot(x, y)
    # axs[1].plot(x, -y)
    for k in [2,4,8]:
      plt.figure(plt_count)
      plt.title("# de clusters = %i | Desvio padrão = %0.1f"%(k,std))
      arr = create_4norm(std)
      results, cluster = v_Kmeans(arr,k)
      for n in range(k):
        indices = [i for i, x in enumerate(cluster) if x == n]
        points = list(zip(*list(map(lambda x,y: [x,y],*[list(itemgetter(*indices)(arr[0])),list(itemgetter(*indices)(arr[1]))]))))
        # plt.subplot(2,1,plt_count)
        plt.scatter(*points)
      plt_count+=1
  plt.show()