from heapq import heapify, heappop, heappush
from importlib import import_module
from turtle import color
from PIL import Image
import numpy as np
from pip import main
from matplotlib import pyplot as plt
from matplotlib import image
from bresenham import bresenham as line_points
import networkx as nx
import time
occ_grid_map = Image.open('occupancy_map.png')
occ_img = (np.asarray(occ_grid_map)>0).astype(int)
x_len = len(occ_img)
y_len = len(occ_img[0])

def vertices(graph):
    #PARAM : A BINARY GRAPH
    #RETURN : A DICTIONARY WITH COORDINATES AS KEY PAIRS AND OCCUPANCY DATA AS THEIR VALUE
    vertices = {}
    for row in range(0,len(graph)):
        for col in range(0,len(graph[0])):
            vertices[(row,col)] = graph[row][col]
    return vertices

def calculate_distance(s,g):
    #Euclidian distance to calculate the H value.
    dx1,dy1 = s
    dx2,dy2 = g
    h = ((dx2-dx1)**2 + (dy2-dy1)**2)**0.5
    return round(h,2)

def uniform_sampling(vertex):
    while True:
        x = int(np.random.uniform(0,x_len))
        y = int(np.random.uniform(0,y_len))
        if vertex[(x,y)] == 1:
            return x,y

def straight_line_planner(vertex,v1,v2):
    x1,y1 = v1
    x2,y2 = v2
    all_points = list(line_points(x1,y1,x2,y2))
    for n in all_points:
        if vertex[n] == 0:# CHECK IF ANY OF THE POINTS IN THE LIST ARE OCCUPIED
            return False
    return True

def addvertex(graph,vnew,dmax):
    #PARAM : Graph, vertex and maximum distance of the line.
    #RETURN : Null
    vox = vertices(occ_img)
    graph.add_node(graph.number_of_nodes()+1,pos=vnew)
    temp =graph.number_of_nodes()
    nodes = list(graph.nodes)
    for v in nodes:
        curr_edge = graph.nodes[v]['pos']
        if straight_line_planner(vox,curr_edge,vnew) and calculate_distance(curr_edge,vnew) <= dmax:
            graph.add_edge(temp,v,weight = calculate_distance(curr_edge,vnew))

def PRM(graph,dmax=75,N=2500):
    #PARAM : Graph,maximum distance of the line, number of iterations.
    #RETURN : Graph
    samp = vertices(occ_img)
    for k in range(0,N):
        # print(k)
        vex = uniform_sampling(samp)
        addvertex(graph,vex,dmax)
    return graph

if __name__ == "__main__":
    '''THIS CODE HAS NOT BEEN OPTIMISED,HENCE THERE ARE SOME RECURRSIVE INSTRUCTIONS IN THE CODE 
    WHICH INCRESES THE COMPPUTATIONAL TIME SIGNIFICANTLY. ONE OPTIMISATION TO BE DONE IS THE IMPLEMENTATION
    OF CLASS IN THIS CODE'''
    G = nx.Graph()
    start_pos = (640, 140)
    goal_pos = (350, 400)
    G.add_node('start',pos= start_pos)
    G.add_node('goal',pos= goal_pos)
    x = PRM(G)
    path = nx.astar_path(x,'start','goal')
    print(len(path))
    data = image.imread('occupancy_map.png')
    fig, ax = plt.subplots()
    data = ax.imshow(data)
    edge = []
    for (e1,e2) in x.edges:
        plt.plot((x.nodes[e1]['pos'][1],x.nodes[e2]['pos'][1]),(x.nodes[e1]['pos'][0],x.nodes[e2]['pos'][0]),linewidth = 0.9,color = 'blue')
    for n in path:
        edge.append(x.nodes[n]['pos'])
    edge = np.array(edge)
    print(len(edge))
    start_x, start_y = x.nodes['start']['pos']
    goal_x, goal_y = x.nodes['goal']['pos']
    plt.plot(edge[:, 1], edge[:, 0], 'r')
    plt.plot(goal_y, goal_x, 'go',color = 'black')
    plt.plot(start_y, start_x, 'ro',color = 'black')
    plt.title('PRM output for d = 75 and N = 400 samples')
    plt.show()