import numpy as np

def plot_cube(structure):
    Z = np.zeros([8, 3])
    Z[0:7, :] = np.array(structure[0:7])
    Z[7, :] = np.array([Z[6, 0], Z[4, 1], Z[5, 2]])
    verts = [[Z[0], Z[1], Z[2], Z[3]],
            [Z[4], Z[5], Z[6], Z[7]],
            [Z[0], Z[1], Z[7], Z[4]],
            [Z[2], Z[3], Z[5], Z[6]],
            [Z[1], Z[2], Z[6], Z[7]],
            [Z[4], Z[7], Z[1], Z[0]]]
    return verts

def plot_structure(pannel):
    a = pannel
    Z = np.zeros([6, 3])
    Z[0, :] = np.array([0, 0, 0])
    Z[1, :] = np.array([0, 0, pannel[11, 2]])
    Z[2, :] = np.array(pannel[10])
    Z[3, :] = np.array([0, pannel[6][1], 0])
    Z[4, :] = np.array(pannel[5])
    Z[5, :] = np.array([pannel[1][0], 0, 0])
    verts = [[Z[0], Z[1], Z[2], Z[3]],
            [Z[0], Z[1], Z[4], Z[5]]]
    return verts

def plot_triangle(structure):
    Z = np.zeros([4, 3])
    Z[0:4, :] = np.array(structure[0:4])
    verts = [[Z[0], Z[1], Z[2]],
            [Z[0], Z[1], Z[3]],
            [Z[3], Z[0], Z[2]],
            [Z[0], Z[2], Z[3]]]
    return verts