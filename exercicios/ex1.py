from numpy.random import normal
from numpy import arange, mean, std, zeros, savetxt, meshgrid
from math import pi, exp, sqrt
from scipy import stats as sts
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

# função de estimativa de densidade
def fnorma1var (x,m,r):
  return ((1/(sqrt(2*pi*r*r)))*exp(-0.5 * ((x-m)/(r))^2))

def pdf2var (x,y,u1,u2,s1,s2,p):
  try:
    if p == 0:
      d = 2*pi*s1*s2
      A = -1/2
      B3 = 0
    else: 
      d = 2*pi*s1*s2*sqrt(1 - (p**2))
      A = (-1)*(1/(2*(1-p**2)))
      B3 = (2*p*(x - u1)*(y - u2))/(s1*s2)
    B1 = ((x - u1)**2.0)/s1**2.0
    B2 = ((y - u2)**2.0)/s2**2.0
    return (
      (1/d)*exp(A*(B1 + B2 - B3))
    )
  except ValueError:
    # print (e)
    return 0

if __name__ == "__main__":
  
  # montando as distribuições
  std_d = 0.6
  
  dist_1_x = normal(2,std_d, 200)
  dist_1_y = normal(2,std_d, 200)
  dist_2_x = normal(4,std_d, 200)
  dist_2_y = normal(4,std_d, 200)
  
  dist_1 = (dist_1_x,dist_1_y)
  dist_2 = (dist_2_x,dist_2_y)
  
  # classificador
  
  theta = 3
  x_range = arange(0,6,0.01)
  y_range = 6*(x_range>theta)
  
  c_range=(x_range,y_range)
  
  # calculando media e desvio padrão
  u1 = mean(dist_1, axis=1)
  u2 = mean(dist_2, axis=1)
  s1 = std(dist_1, axis=1)
  s2 = std(dist_2, axis=1)
  
  # estimativa das densidades por ponto do grid
  seqi=arange(0.06,6,0.06)
  seqj=arange(0.06,6,0.06)
  m_size = [len(seqi),len(seqj)]
  print (m_size)
  M0 = zeros(m_size)
  M1 = zeros(m_size)
  M2 = zeros(m_size)
  
  f = 0
  
  ci = 0
  for i in seqi:
    ci += 1 
    cj = 0
    for j in seqj:
      cj += 1
      M1[ci-1,cj-1] = pdf2var(i,j,*u1,*s1,f)
      M2[ci-1,cj-1] = pdf2var(i,j,*u2,*s2,f)
  
  savetxt('M1.csv', M1, delimiter=';')
  
  # fig_1 = plt.figure(1)
  # plt.scatter(*dist_1)
  # plt.scatter(*dist_2)
  # plt.plot(*c_range)
  
  seqi, seqj = meshgrid(seqi,seqj)
  fig_2 = plt.figure(2)
  ax = fig_2.gca(projection='3d')
  surf_1 = ax.plot_surface(seqi,seqj,M1+M2, cmap=cm.coolwarm)
  ax.set_title('Superficie p=%d'%(f))
  
  cntr, dnd = plt.subplots()
  CS = dnd.contour(seqi,seqj,M1+M2)
  dnd.clabel(CS, inline =1)
  dnd.set_title('Contorno p=%d'%(f))
  plt.show()