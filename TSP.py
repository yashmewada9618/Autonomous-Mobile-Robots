from itertools import permutations

a = 3 #VALUE OF ALPHA
V = 4 # NUMBER OF NODES IN THE GIVEN GRAPH
all_cost = [] # EMPTY LIST OF ALL THE COST TO GO
graph = [[0, 2, 1, 1],
        [2, 0, a, 2],
        [1, a, 0, 2],
        [1, 2, 1, 0]]
    
def get_cost(graph,row,col):
    return graph[row][col]

def TSP(graph, s):
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)
    #PERMUTATION OF THE NODES OF THE GRAPH ARE [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    for i in permutations(vertex):
        current_pathweight = 0\
        # MAKE START AS PREVIOUS PATH AND THEN ITERATE OVER THE GIVEN NODES
        #THEN MAKE THE LAST VISITED PATH AS THE START PATH. THIS IS KIND OF MEMOIZE BY KEEPING A NOTE OF PREVIOUSLY VISITED NODE.
        previous_path = s
        for j in i:
            current_pathweight += get_cost(graph,previous_path,j)
            previous_path = j
        current_pathweight += get_cost(graph,previous_path,s) 
        all_cost.append(current_pathweight)
    return all_cost
if __name__ == "__main__":
    s = 0
    print(TSP(graph, s))