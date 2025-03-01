import numpy as np
import matplotlib.pyplot as plt
import cdd
import sys
import numerics
from importlib import reload
from sklearn.manifold import TSNE

def make_index_mat(P):
    M = np.ones((P, P)) * -1
    idx = np.triu_indices(P, k=1)
    n = P*(P-1)//2
    M[idx] = np.arange(n)
    M.T[idx] = np.arange(n)
    return M

def make_disp_str(mask, return_mat=False):
    n = len(mask)
    P = int(0.5*(1 + np.sqrt(1. + 8*n)))
    M = make_index_mat(P)
    disp = np.zeros((P, P)).astype(str)
    disp[:] = ' · '
    for i in range(P):
        for j in range(P):
            if i==j: continue
            if mask[int(M[i,j])]:
                disp[i,j] = ' X '
    if return_mat:
        return (disp == ' X ').astype(float)
    string = '\n'.join([''.join(disp[i]) for i in range(P)])
    return string
    
def make_constraint_mat(prms):
    P = len(prms)
    #A = numerics.compute_psi_inv(1./prms)
    n = P*(P-1)//2
    C = np.zeros((P, n))
    M = make_index_mat(P)
    for i in range(P):
        for j in range(n):
            if j in M[i]:
                col_idx = np.argmax(M[i] == j)
                lower = i > col_idx
                C[i,j] = prms[i]/prms[col_idx] if lower else 1.
    return C

def compute_vertices(prms):
    A = numerics.compute_psi_inv(1./prms)
    P = len(A)
    C = make_constraint_mat(prms)
    rows1 = np.concatenate((-A[:, None], C), axis=1)
    rows1 = [list(r) for r in rows1]
    C_cdd = cdd.Matrix(rows=rows1, linear=True, number_type='float')
    n = P*(P-1)//2
    rows2 = np.concatenate((np.zeros((n, 1)), np.eye(n)), axis=1)
    rows2 = [list(r) for r in rows2]
    C_cdd.extend(rows=rows2, linear=False)
    C_cdd.rep_type = cdd.RepType.INEQUALITY
    C_cdd.canonicalize()
    try:
        poly = cdd.Polyhedron(C_cdd)
        vertices = poly.get_generators()
        vertices = np.array(vertices[:])
        if vertices.ndim > 1:
            mask_nz = np.all(np.abs(vertices[:,1:]) > 1e-6, axis=0).astype(int)
            mask_z = np.all(np.abs(vertices[:,1:]) <= 1e-6, axis=0).astype(int)
            success = True
        else:
            mask_nz, mask_z = np.zeros((2, n))
            success = True
    except RuntimeError as m:
        success = False
    if success:
        return mask_nz, mask_z, vertices, success
    else:
        return None, None, None, success