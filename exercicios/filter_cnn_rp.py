from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv

def my_ReLU(arr):
  for row in range(len(arr)):
    for col in range(len(arr[row])):
      pixel = arr[row][col] if arr[row][col] > 0 else 0
      arr[row][col] = pixel
  return arr

def my_Max_Pooling():
  pass

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

print(my_ReLU(filter_vert))