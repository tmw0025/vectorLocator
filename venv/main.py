from typing import List, Any
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as cm
from PIL import Image, ImageChops
from scipy import ndimage as ndi
from skimage import feature
import networkx as nx
import sys
import heapq
from queue import LifoQueue


class node:
    def __init__(self, color, dist, prev, next, visited):
        self.color = color
        self.dist = dist
        self.prev = prev
        self.next = next
        self.visited = visited


G = {}
maxRows = 0
maxCols = 0


def length(S, U):
    x1 = S[0]
    x2 = U[0]
    y1 = S[1]
    y2 = U[1]
    return math.sqrt((x2-x1)**2+(y2-y1)**2)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def vertices(S):
    i = S[0]
    j = S[1]
    v = []
    if i == 0:
        if j == 0:
            v.append(((i+1), j))
            v.append(((i+1), (j+1)))
            v.append((i, (j+1)))
        elif j == maxCols:
            v.append(((i+1), j))
            v.append(((i+1), (j-1)))
            v.append((i, (j-1)))
        else:
            v.append((i, (j-1)))
            v.append(((i+1), (j-1)))
            v.append(((i+1), j))
            v.append(((i+1), (j+1)))
            v.append((i, (j+1)))
    elif i == maxRows:
        if j == 0:
            v.append(((i - 1), j))
            v.append(((i - 1), (j + 1)))
            v.append((i, (j + 1)))
        elif j == maxCols:
            v.append(((i - 1), j))
            v.append(((i - 1), (j - 1)))
            v.append((i, (j - 1)))
        else:
            v.append(((i - 1), j))
            v.append(((i - 1), (j - 1)))
            v.append((i, (j - 1)))
            v.append(((i - 1), (j + 1)))
            v.append((i, (j + 1)))
    elif j == 0:
        v.append(((i - 1), j))
        v.append(((i + 1), (j + 1)))
        v.append(((i + 1), j))
        v.append(((i - 1), (j + 1)))
        v.append((i, (j + 1)))
    elif j == maxCols:
        v.append(((i - 1), j))
        v.append(((i - 1), (j - 1)))
        v.append((i, (j - 1)))
        v.append(((i + 1), (j - 1)))
        v.append(((i + 1), j))
    else:
        v.append(((i - 1), (j + 1)))
        v.append(((i - 1), j))
        v.append(((i - 1), (j - 1)))
        v.append(((i + 1), j))
        v.append(((i + 1), (j - 1)))
        v.append((i, (j - 1)))
        v.append(((i + 1), (j + 1)))
        v.append((i, (j + 1)))
    return v


def dijkstra(G, S):
    G[S].dist = 0
    Q = []
    prev = []
    graphSize = 1
    for each in G.keys():
        if G[each].color == 1:
            heapq.heappush(Q, (G[each].dist, each))
    while Q:
        u = heapq.heappop(Q)
        for e in vertices(u[1]):
            print(e, end='\tfrom\t')
            print(u[1])
            alt = G[u[1]].dist + length(u[1], e)
            if alt < G[e].dist and G[e].visited != 1 and G[e].color == 1:
                G[e].visited = 1
                index = Q.index((G[e].dist, e))
                G[e].dist = alt
                prev.append(u[1])
                prev.sort()
                graphSize += 1
                Q[index] = (alt, e)
            heapq.heapify(Q)
    return prev


img = mpimg.imread('ProfFarns.png')
gray = rgb2gray(img)
# Compute the Canny filter for two values of sigma
edges1 = feature.canny(gray)
edges2 = feature.canny(gray, sigma=0.75)

print(edges2.shape)
rows = edges2.shape[0]
maxRows = edges2.shape[0]
cols = edges2.shape[1]
maxCols = edges2.shape[1]
passFlag = 0
for i in range(0, cols):
    for j in range(0, rows):
        if edges2[j, i] == 1:
            print("Index: %s\tvalue: %d" % ((i, j), edges2[j, i]))
            firstRow = i
            firstCol = j
            passFlag = 1
            break
    if passFlag == 1:
        break

#for i in range(firstRow, 0, -1):
    # print((i, firstCol), end=' ')
    # print(edges2[firstCol, i])
    #edges2[firstCol, i] = 1
    # print(edges2[firstCol, i])

for i in range(0, cols, 1):
    for j in range(0, rows, 1):
        G[(i, j)] = node(edges2[j, i], sys.maxsize, -1, -1, 0)

edges2 = ndi.binary_dilation(edges2, iterations=2)
edges2 = ndi.binary_erosion(edges2, iterations=2)
e1 = Image.fromarray(np.uint8(edges2))
e2 = Image.fromarray(np.uint8(edges2))

prev = dijkstra(G, (firstRow, firstCol))
#e2.convert(mode='RGB')
red = (255, 0, 0)
print("Begin", end='->')
for each in prev:
    e2.putpixel(each, 0)
    print(each, end='->')
print("End")
# display results
e2 = ImageChops.difference(e1, e2)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(gray, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Original image', fontsize=20)


ax2.imshow(e2, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=0.75$', fontsize=20)

fig.tight_layout()

plt.show()

