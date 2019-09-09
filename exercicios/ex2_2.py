from numpy.random import normal, rand
import matplotlib.pyplot as plt
from numpy import mean, std, zeros, arange, meshgrid, concatenate, array
from itertools import compress
from ex1 import pdf2var
from ex2 import create_surface, split_for_train_test,classifier_2v

if __name__ == "__main__":

  points = 300

  std_d_2 = 0.6
  dist_2_x = normal(2,std_d_2, points)
  dist_2_y = normal(-2,std_d_2, points)
  dist_3_x = normal(-2,std_d_2, points)
  dist_3_y = normal(2,std_d_2, points)
  dist_p1 = [concatenate((dist_2_x,dist_3_x)),concatenate((dist_2_y,dist_3_y))]

  dist_4_x = normal(-2,std_d_2, points)
  dist_4_y = normal(-2,std_d_2, points)
  dist_1_x = normal(2,std_d_2, points)
  dist_1_y = normal(2,std_d_2, points)
  dist_p2 = [concatenate((dist_1_x,dist_4_x)),concatenate((dist_1_y,dist_4_y))]

  msk = rand(points*2) < 0.9

  train_x, test_x = split_for_train_test(dist_p1[0],msk)
  train_y, test_y = split_for_train_test(dist_p1[1],msk)

  train_1 = [train_x,train_y]
  test_1 = [test_x,test_y]

  train_x, test_x = split_for_train_test(dist_p2[0],msk)
  train_y, test_y = split_for_train_test(dist_p2[1],msk)
  
  train_2 = [train_x,train_y]
  test_2 = [test_x,test_y]


  u1 = mean(train_1, axis=1)
  u2 = mean(train_2, axis=1)
  s1 = std(train_1, axis=1)
  s2 = std(train_2, axis=1)

  correlacao = 0

  rslt = []
  rslt.append(classifier_2v(test_1,u1,u2,s1,s2,correlacao))
  rslt.append(classifier_2v(test_2,u1,u2,s1,s2,correlacao))

  lim_inf = -4
  lim_ext = 4
  discrt = 0.05

  seqi=arange(lim_inf,lim_ext,discrt)
  seqj=arange(lim_inf,lim_ext,discrt)

  grid = create_surface(u1, u2, s1, s2, correlacao, seqi, seqj)

  fig, ax = plt.subplots()
  ax.scatter(*dist_p1)
  ax.scatter(*dist_p2)
  ax.set_title('Distribuições no espaço R2')
  
  fig2, ax2 = plt.subplots()
  ax2.scatter(*train_1)
  ax2.scatter(*train_2)
  ax2.contour(seqi, seqj, grid)
  ax2.set_title('Contorno de separação')

  # plt.plot(*c_range)
  
  plt.show()