from numpy.random import normal, rand
import matplotlib.pyplot as plt
from numpy import mean, std, zeros, arange, meshgrid
from itertools import compress
from ex1 import pdf2var
# from mpl_toolkits import mplot3d

def classifier (x, y, u1, u2, s1, s2, p):
  v0 = pdf2var(x,y,*u1,*s1,p)
  v1 = pdf2var(x,y,*u2,*s2,p)
  comp = v0/v1
  if comp >= 1: 
    return 1
  else :
    return -1

def classifier_2v(t,u1,u2,s1,s2,p):
  K = []
  for i in range(len(t[0])):
    K.append(classifier(t[0][i], t[1][i],u1,u2,s1,s2,p))
  return K

def create_surface(u1, u2, s1, s2, f, seqi, seqj): #lim_inf=0.06, lim_ext=6, discrt = 0.06):
  m_size = [len(seqi), len(seqj)]
  grid = zeros(m_size)
  ci = 0
  for i in seqi:
    ci += 1
    cj = 0
    for j in seqj:
      cj += 1
      grid[ci-1,cj-1] = classifier(i,j,u1,u2,s1,s2,f)
  
  return grid

def split_for_train_test(arr, percent, axis = 1):
  if axis == 1:
    train = list(compress(arr,percent))
    test = list(compress(arr,~percent))
    return train, test
  else:
    train_rslt = []
    test_rslt = []
    for ax in arr:
      train = list(compress(ax,percent))
      test = list(compress(ax,~percent))
      train_rslt.append(train)
      test_rslt.append(test)
    return train_rslt, test_rslt
    


points = 200
if __name__ == "__main__":
  
  # montando as distribuições
  std_d_1 = 0.8
  std_d_2 = 0.4
  dist_1_x = normal(2,std_d_1, points)
  dist_1_y = normal(2,std_d_1, points)
  dist_2_x = normal(4,std_d_2, points)
  dist_2_y = normal(4,std_d_2, points)

  dist_1 = [dist_1_x,dist_1_y]
  dist_2 = [dist_2_x,dist_2_y]

  # separação dos dados

  msk = rand(points) < 0.9

  train_x, test_x = split_for_train_test(dist_1_x,msk)
  train_y, test_y = split_for_train_test(dist_1_y,msk)

  train_1 = [train_x,train_y]
  test_1 = [test_x,test_y]

  train_x, test_x = split_for_train_test(dist_2_x,msk)
  train_y, test_y = split_for_train_test(dist_2_y,msk)
  
  train_2 = [train_x,train_y]
  test_2 = [test_x,test_y]

  # treinamento

  u1 = mean(train_1, axis=1)
  u2 = mean(train_2, axis=1)
  s1 = std(train_1, axis=1)
  s2 = std(train_2, axis=1)

  correlacao = 0

  # teste

  rslt = []
  rslt.append(classifier_2v(test_1,u1,u2,s1,s2,correlacao))
  rslt.append(classifier_2v(test_2,u1,u2,s1,s2,correlacao))

  # resultados dos testes
  # TODO

  # criando grid de contorno

  lim_inf = 0.06
  lim_ext = 6
  discrt = 0.06

  seqi=arange(lim_inf,lim_ext,discrt)
  seqj=arange(lim_inf,lim_ext,discrt)
  # seqi=arange(0.06,6,0.06)
  # seqj=arange(0.06,6,0.06)

  grid = create_surface(u1, u2, s1, s2, correlacao, seqi, seqj)

  # plotando os gráficos


  seqi, seqj = meshgrid(seqi, seqj)

  fig, ax = plt.subplots()
  ax.scatter(*dist_1)
  ax.scatter(*dist_2)
  ax.set_title('Distribuições no espaço R2')
  fig2, ax2 = plt.subplots()
  ax2.scatter(*train_1)
  ax2.scatter(*train_2)
  ax2.contour(seqi, seqj, grid)
  ax2.set_title('Contorno de separação')
  # plt.plot(*c_range)

  plt.show()