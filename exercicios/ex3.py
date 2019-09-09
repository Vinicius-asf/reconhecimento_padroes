from numpy.random import normal, rand
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import mean, std, zeros, arange, meshgrid, concatenate, array
from itertools import compress
from mpl_toolkits import mplot3d

from ex2 import create_surface, split_for_train_test
from ex1 import pdf2var

def classifier4v(x, y, u, s, p, clss = 0):
  v = []
  for i in range(len(u)):
    v.append(pdf2var(x,y,*u[i],*s[i],p))
  return v.index(max(v))

def surface_4_dist(u, s, corr, seqi, seqj):
  m_size = [len(seqi), len(seqj)]
  grid = zeros(m_size)
  ci = 0
  for i in seqi:
    ci += 1
    cj = 0
    for j in seqj:
      cj += 1
      grid[ci-1,cj-1] = classifier4v(j,i,u,s,corr)
  return grid

if __name__ == "__main__":
  
  points = 100

  std_d_1 = 0.6
  std_d_2 = 0.8
  std_d_3 = 0.2
  std_d_4 = 1

  dist_x = normal(2,std_d_1, points)
  dist_y = normal(2,std_d_1, points)
  dist_p1 = [dist_x,dist_y]

  dist_x = normal(4,std_d_2, points)
  dist_y = normal(4,std_d_2, points)
  dist_p2 = [dist_x,dist_y]

  dist_x = normal(2,std_d_3, points)
  dist_y = normal(4,std_d_3, points)
  dist_p3 = [dist_x,dist_y]

  dist_x = normal(4,std_d_4, points)
  dist_y = normal(2,std_d_4, points)
  dist_p4 = [dist_x,dist_y]

  # separação dos dados

  msk = rand(points) < 0.9

  train_1, test_1 = split_for_train_test(dist_p1,msk,2)
  train_2, test_2 = split_for_train_test(dist_p2,msk,2)
  train_3, test_3 = split_for_train_test(dist_p3,msk,2)
  train_4, test_4 = split_for_train_test(dist_p4,msk,2)

  # treinamento
  u = []
  s = []
  
  u.append(mean(train_1,axis=1))
  u.append(mean(train_2,axis=1))
  u.append(mean(train_3,axis=1))
  u.append(mean(train_4,axis=1))

  s.append(std(train_1, axis=1))
  s.append(std(train_2, axis=1))
  s.append(std(train_3, axis=1))
  s.append(std(train_4, axis=1))

  # definindo grid
  lim_inf = 0.06
  lim_ext = 6
  discrt = 0.06

  seqi=arange(lim_inf,lim_ext,discrt)
  seqj=arange(lim_inf,lim_ext,discrt)
  m_size = [len(seqi),len(seqj)]

  M1 = zeros(m_size)
  M2 = zeros(m_size)
  M3 = zeros(m_size)
  M4 = zeros(m_size)

  
  f = 0
  
  ci = 0
  for i in seqi:
    ci += 1 
    cj = 0
    for j in seqj:
      cj += 1
      M1[ci-1,cj-1] = pdf2var(i,j,*u[0],*s[0],f)
      M2[ci-1,cj-1] = pdf2var(i,j,*u[1],*s[1],f)
      M3[ci-1,cj-1] = pdf2var(i,j,*u[2],*s[2],f)
      M4[ci-1,cj-1] = pdf2var(i,j,*u[3],*s[3],f)

  grid = surface_4_dist(u, s, 0, seqi, seqj)

  fig, ax = plt.subplots()
  ax.scatter(*dist_p1)
  ax.scatter(*dist_p2)
  ax.scatter(*dist_p3)
  ax.scatter(*dist_p4)
  ax.contour(seqi, seqj, grid)

  seqi, seqj = meshgrid(seqi,seqj)
  fig_2 = plt.figure(2)
  ax = fig_2.gca(projection='3d')
  surf_1 = ax.plot_surface(seqi,seqj,grid, cmap=cm.coolwarm)

  fig_3 = plt.figure(3)
  ax = fig_3.gca(projection='3d')
  # surf_2 = ax.plot_surface(seqi,seqj,M1+M2+M3+M4, cmap=cm.coolwarm)
  surf_2 = ax.plot_wireframe(seqi, seqj, M1+M2+M3+M4, rstride=10, cstride=10, cmap=cm.coolwarm)
  ax.set_zlim(-0.1, 1)

  fig_4 = plt.figure(4)
  ax = fig_4.gca(projection='3d')
  surf_2 = ax.plot_surface(seqi,seqj,M1+M2+M3+M4, cmap=cm.coolwarm)
  # surf_2 = ax.plot_wireframe(seqi, seqj, M1+M2+M3+M4, rstride=10, cstride=10, cmap=cm.coolwarm)
  ax.set_zlim(-0.1, 1)

  fig_5, ax = plt.subplots()
  ax.scatter(*dist_p1)
  ax.scatter(*dist_p2)
  ax.scatter(*dist_p3)
  ax.scatter(*dist_p4)

  plt.show()