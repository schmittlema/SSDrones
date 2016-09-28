import numpy as np
import copy as cp
import queue as qu

"""matrix = np.zeros((8,5),dtype=object)
matrix[0][0] = 3
matrix[4][0] = 2
matrix[4][4] = 1
matrix[0][4] = 2
matrix[4][1] = 2
matrix[3][1] = 2
matrix[2][1] = 2
"""

def make_matrix(matrix):
    mat = cp.copy(matrix)
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 2:
                mat[i][j] = []
            else:
                mat[i][j] = [0,0,i,j]
    return mat

def valid(point):
    if point != []:
        if point[0] == 0:
            return True
    return False
        

def next_moves(p,mat):
    result = []
    point = [p[2],p[3]]
    if point[0] - 1 >= 0:
        if valid(mat[point[0]-1][point[1]]):
            result.append(mat[point[0]-1][point[1]])
    if point[0] + 1 < len(mat):
        if valid(mat[point[0]+1][point[1]]):
            result.append(mat[point[0]+1][point[1]])
    if point[1] - 1 >= 0:
        if valid(mat[point[0]][point[1] - 1]):
            result.append(mat[point[0]][point[1]-1])
    if point[1] + 1 < len(mat[0]):
        if valid(mat[point[0]][point[1]+1]):
            result.append(mat[point[0]][point[1]+1])
    return result

def bfs(matrix,point):
    mat = make_matrix(matrix)
    result = cp.copy(mat)
    result[point[0]][point[1]] = 0
    mat[point[0]][point[1]] = [1,0,point[0],point[1]]
    q = qu.Queue()
    q.put([1,0,point[0],point[1]])

    while not q.empty():
        u = q.get()
        moves = next_moves(u,mat) 
        for v in moves:
            if not v[0]:
                v[0] = 1
                v[1] = u[1] + 1
                result[v[2]][v[3]] = v[1]
                q.put(v)
    return result


