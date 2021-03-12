import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from imageGraph import *
from disjointSet import *

#read the Image and apply gaussian filter
img = cv2.imread("image/tom.jpg")
img = np.asarray(img,dtype='float')
img = cv2.GaussianBlur(img,(3,3),0.5)
height = img.shape[0]
width = img.shape[1]

#threshold f()
def tau(k,C):
    return (k/C)

def random_color():
    return int(random.randint(0,255))

def segment(graph,vertices,k=300):
    threshold = [tau(k,1)] * vertices
    ds = DisjointSet(vertices)
    sort_graph = sorted(graph, key=lambda item: item[2])
    weight = lambda edge:edge[2]
    for edge in sort_graph:
        xp = ds.find(edge[0])
        yp = ds.find(edge[1])
        w = weight(edge)
        if xp!=yp and w<=threshold[xp] and w<=threshold[yp]:
            ds.union(xp,yp)
            p = ds.find(xp)
            threshold[p] = weight(edge) + tau(k,ds.size[p])
    return ds

def remove_small_component(ds, graph, min_size):
    sort_graph = sorted(graph, key=lambda item: item[2])
    for edge in sort_graph:
        u = ds.find(edge[0])
        v = ds.find(edge[1])

        if u != v and  ds.size[u] < min_size or ds.size[v] < min_size:
                ufset.union(u, v)

    return ufset

#generate Image
def generate_image(ds,width,height):
    generated_image = np.zeros((height, width, 3), np.uint8)
    pixel_color = [(random_color(), random_color(), random_color()) for i in range(height * width)]
    for y in range(height):
        for x in range(width):
            index = ds.find(y * width + x)
            generated_image[y, x] = pixel_color[index]
    return generated_image

#run
graph = image_graph(img,height,width)
ds = segment(graph,height*width,k=300)
ds = remove_small_component(ds,graph,50)
generated_image = generate_image(ds,width,height)
plt.imshow(generated_image)
plt.show()




