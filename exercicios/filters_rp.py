from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv

# from scipy import ndimage

# faces = fetch_olivetti_faces()

data = fetch_olivetti_faces().data
data_int = np.rint(data*255).astype(int)
tgt = fetch_olivetti_faces().target
images_b = fetch_olivetti_faces().images
images = np.rint(images_b*255).astype(int)

fig_num = 1
face = 1

plt.figure(fig_num)
fig_num+=1
plt.title("original")
plt.imshow(images[face], cmap=plt.cm.gray, interpolation='nearest')
plt.xticks(())
plt.yticks(())
plt.axis('off')

lx, ly = images[face].shape
tf = 1


filter_edge = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
filter_vert = [[1,2,1],[0,0,0],[-1,-2,-1]]
filter_horz = [[1,0,-1],[2,0,-2],[1,0,-1]]
filter_sharp = [[0,-1,0],[-1,5,-1],[0,-1,0]]
filters = [
  {"filter": filter_edge, "name": "edge"},
  {"filter":filter_vert,  "name": "vertical"},
  {"filter":filter_horz,  "name":"horizontal"},
  {"filter":filter_sharp, "name":"sharper"}
]

for fltr in filters:
  rslt = np.zeros((lx-2,ly-2), dtype=int)
  for x in range(1, lx-tf):
    for y in range(1, ly-tf):
      rslt[x-1,y-1]=np.sum(np.multiply(images[face][x-1:x+2,y-1:y+2],fltr["filter"])) 
  
  plt.figure(fig_num)
  fig_num+=1
  plt.title(fltr["name"])
  plt.imshow(rslt, cmap=plt.cm.gray, interpolation='nearest')
  plt.xticks(())
  plt.yticks(())
  plt.axis('off')

plt.show()