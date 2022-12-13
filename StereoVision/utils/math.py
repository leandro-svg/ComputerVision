import numpy as np
    
    
def SVD(P):
    u, s, vh = np.linalg.svd(P)
    v = vh.T
    M = v[:, -1].T
    M = np.reshape(M, (3,4))
    norm_3 = np.linalg.norm(M[2,:])
    M = M/norm_3
    return M

def SVD_coord(A):
    u, s, vh = np.linalg.svd(A)
    v = vh.T
    pred_world = v[:, -1]
    norm_3 = (pred_world[-1])
    pred_world = (pred_world/norm_3)
    return pred_world