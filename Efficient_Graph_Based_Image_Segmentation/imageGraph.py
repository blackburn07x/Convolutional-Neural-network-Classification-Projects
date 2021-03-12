import numpy as np
#buid graph
def weight(img,x1,y1,x2,y2): #f() to find the
    red = np.power(img[:,:,0][y1,x1] - img[:,:,0][y2,x2], 2)
    green = np.power(img[:,:,1][y1,x1] - img[:,:,1][y2,x2], 2)
    blue = np.power(img[:,:,2][y1,x1] - img[:,:,2][y2,x2], 2)
    return np.sqrt(red + green + blue)

def image_graph(img,height,width):
    graph = []
    for y in range(height):
        for x in range(width):
            if x <width-1:
                m1 = y * width + x
                m2 = y * width + (x + 1)
                graph.append((m1, m2, weight(img, x, y, x + 1, y)))
            if y <height-1:
                m1 = y * width + x
                m2 = (y + 1) * width + x
                graph.append((m1, m2, weight(img, x, y, x, y - 1)))

    return  graph