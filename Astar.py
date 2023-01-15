from heapq import heapify, heappop, heappush
from PIL import Image
import numpy as np
from pip import main
from matplotlib import pyplot as plt
from matplotlib import image

occ_grid_map = Image.open('occupancy_map.png')
occ_img = (np.asarray(occ_grid_map)>0).astype(int)

def calculate_distance(s,g):
    #Euclidian distance to calculate the H value.
    dx1,dy1 = s
    dx2,dy2 = g
    h = ((dx2-dx1)**2 + (dy2-dy1)**2)**0.5
    return h
 
def get_occupancy_neighbour(graph,position):
    #THIS FUNCTION RETURNS THE SET OF ALL THE UNOCCUPIED NEIGHBOURS OF THE GIVEN POSITION
    #PARAM : BINARY OCCUPANCY GRID MAP AND THE POSITION WHOES NEIGHBORS IS NEEDED
    #RETURN : A LIST OF UNOCCUPIED NEIGHBORS
    neighbours =[(1, 0),(0, 1),(-1, 0),
            (0, -1),(1, 1),(-1, 1),(-1, -1),
            (1, -1)]
    neighbour_pos = []
    for x,y in neighbours:
        neighbour_x = position[0] + x
        neighbour_y = position[1] + y
        if graph[(neighbour_x,neighbour_y)] == 1:
            neighbour_pos.append((neighbour_x,neighbour_y))
    return neighbour_pos

def recover_path(previous_path,current_path,pred):
    ''' This function takes previous,current and predecessor as its input and 
    outputs the most optimal path'''
    #PARAM : PARENT PATH,CURRENT PATH AND PREDECESSOR MAP
    #RETURN : AN OPTIMAL PATH BY ITERATING OVER ITS PARENTS
    optimal_path = [current_path]
    temp_ver = current_path
    while temp_ver != previous_path:
        temp_ver = pred[temp_ver]
        optimal_path.append(temp_ver)
    return optimal_path

def vertices(graph):
    #PARAM : A BINARY GRAPH
    #RETURN : A DICTIONARY WITH COORDINATES AS KEY PAIRS AND OCCUPANCY DATA AS THEIR VALUE
    vertices = {}
    for row in range(0,len(graph)):
        for col in range(0,len(graph[0])):
            vertices[(row,col)] = graph[row][col]
    return vertices

def astarsearch(graph, start, goal):
    #PARAM : VERTICES OF GRAPH AS DICTIONARY,START POSITION AND GOAL POSITION
    #RETURN : OPPTIMAL PATH FROM START TO GOAL.
    cost_to_go = {} 
    est_total_cost = {}
    pred = {}
    voxal = vertices(graph)
    if not voxal[start] or not voxal[goal]:
        return Exception("The start or goal positions are occupied")
    for every_vertex in voxal:
        cost_to_go[every_vertex] = float('inf')
        est_total_cost[every_vertex] =float('inf') 
    cost_to_go[start] = 0
    est_total_cost[start] = calculate_distance(start,goal)
    Q = [(calculate_distance(start,goal),start)]
    while Q:
        newu,vox = heappop(Q)
        if vox == goal:
            return recover_path(start,goal,pred)
        for each_neighbor in get_occupancy_neighbour(voxal,vox):
            pvi = cost_to_go[vox] + calculate_distance(vox,each_neighbor)
            if pvi < cost_to_go[each_neighbor]:
                pred[each_neighbor] = vox
                cost_to_go[each_neighbor] = pvi
                est = est_total_cost[each_neighbor]
                est_total_cost[each_neighbor] = pvi + calculate_distance(each_neighbor,goal)
                if(est,each_neighbor) in Q:
                    Q.remove((est,each_neighbor))
                heappush(Q,(est_total_cost[each_neighbor],each_neighbor))

if __name__ == "__main__":
    start_pos = (500, 140)
    goal_pos = (350, 500)
    path = astarsearch(occ_img,start_pos,goal_pos)
    print(len(path)) #OVERALL COST OF THE PATH IS 725
    start_x, start_y = path[-1]
    goal_x, goal_y = path[0]
    path_arr = np.array(path)
    print(path)
    data = image.imread('occupancy_map.png')
    fig, ax = plt.subplots()
    data = ax.imshow(data)
    plt.plot(path_arr[:, 1], path_arr[:, 0], 'r')
    plt.plot(goal_y, goal_x, 'go',color = 'blue')
    plt.plot(start_y, start_x, 'ro',color = 'blue')
    plt.title('A* Search for start: (635,140) and goal: (350,400)')
    plt.show()