import numpy as np
import scipy as sp
import numpy.linalg as la

from numba import njit
from numba.typed import List


@njit
def calc_def_grad(vert_list, pos, disp, ref_vert=None, ref_pos=np.zeros(0, np.float64), ref_disp=np.zeros(0, np.float64), weighted=False, weights=np.zeros(0, np.float64)):
        
    DIM = 3
    NV = len(disp)//DIM
    
    if ref_vert is not None:
        ref_pos = pos[DIM*ref_vert:DIM*ref_vert+DIM]
        ref_disp = disp[DIM*ref_vert:DIM*ref_vert+DIM]
    
    X = np.zeros((DIM, DIM), np.float64)
    Y = np.zeros((DIM, DIM), np.float64)

    for i in range(len(vert_list)):
        
        vi = vert_list[i]
        
        bvec = pos[DIM*vi:DIM*vi+DIM] - ref_pos
        du = disp[DIM*vi:DIM*vi+DIM] - ref_disp
        
        if weighted:
            X += weights[i]*np.outer(bvec, bvec)
            Y += weights[i]*np.outer(bvec + du, bvec)
        else:
            X += np.outer(bvec, bvec)
            Y += np.outer(bvec + du, bvec)
    
    F = Y@la.inv(X)
    
    return F

@njit
def decompose_def_grad(F, linear=True):
    
    DIM = 3
    
    if linear:
        R = np.identity(DIM) + 0.5*(F - F.T)
        U = 0.5*(F + F.T)
                
    else:
        
        C = F.T @ F;
        
        evals, evecs = la.eigh(C)
    
        U = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    
        R = F @ la.inv(U);
        
    return R, U

@njit
def calc_global_motion(vert_list, pos, disp):
    
    DIM = 3
     
    xcm = np.zeros(DIM, np.float64)
    ucm = np.zeros(DIM, np.float64)
        
    for vi in vert_list:
        xcm += pos[DIM*vi:DIM*vi+DIM]
        ucm += disp[DIM*vi:DIM*vi+DIM]
        
    xcm /= len(vert_list)
    ucm /= len(vert_list)
        
    F = calc_def_grad(vert_list, pos, disp, ref_pos=xcm, ref_disp=ucm)
    
    return xcm, ucm, F

@njit 
def subtract_global_motion(pos, xcm, ucm, R):

    DIM = 3
    NV = len(pos)//DIM
    
    pos_new = np.zeros_like(pos)
    
    for vi in range(NV):
        pos_new[DIM*vi:DIM*vi+DIM] = R.T@(pos[DIM*vi:DIM*vi+DIM] - xcm - ucm)
        
    return pos_new
    
        

@njit   
def calc_rmsd(ref_vert, vert_list, pos, disp, linear=False, weighted=False, weights=np.zeros(0, np.float64)):
    
    DIM = 3
    
    F = calc_def_grad(vert_list, pos, disp, ref_vert=ref_vert, weighted=weighted, weights=weights)
        
    R, U = decompose_def_grad(F, linear=linear)
        
    ref_pos = pos[DIM*ref_vert:DIM*ref_vert+DIM]
    ref_disp = disp[DIM*ref_vert:DIM*ref_vert+DIM]
    
    
    rmsd = 0.0
    
    for i in range(len(vert_list)):
        
        vi = vert_list[i]
        
        bvec = pos[DIM*vi:DIM*vi+DIM] - ref_pos
        du = disp[DIM*vi:DIM*vi+DIM] - ref_disp

        if weighted:
            rmsd += weights[i]*la.norm(R@bvec - (bvec+du))**2
        else:
            rmsd += la.norm(R@bvec - (bvec+du))**2
   
    if weighted:
        rmsd /= (weights.sum()-1)
    else:
        rmsd /= (len(vert_list)-1)
   
    return np.sqrt(rmsd)


@njit
def calc_local_rmsd(pos, disp, max_dist, linear=False, weighted=False):
    
    DIM = 3
    
    NV = len(disp)//DIM
    
    # construct neighbor grid
    # sort vertices into grid with cells of size max_dist x max_dist x max_dist
    
    Lmin = np.zeros(DIM, np.float64)
    Lmax = np.zeros(DIM, np.float64)
    for i in range(DIM):
        Lmin[i] = pos.reshape((NV, DIM))[:, i].min()
        Lmax[i] = pos.reshape((NV, DIM))[:, i].max()
        
#     print(Lmin)
           
    # make box 1% bigger so that atoms at boundary don't result in indexing errors
    L =  1.01*(Lmax - Lmin)
    Ncells = np.floor(L / max_dist).astype(np.int32)
    
#     print(Ncells)
    
    # create grid 
    neighbor_grid = [[[[np.int32(x) for x in range(0)] for iz in range(Ncells[2])] for iy in range(Ncells[1])] for ix in range(Ncells[0])]

    # sort vertices into grid
    for vi in range(NV):
        IX = np.floor((pos[DIM*vi:DIM*vi+DIM] - Lmin) / L * Ncells).astype(np.int32)
        neighbor_grid[IX[0]][IX[1]][IX[2]].append(np.int32(vi))
        
    # calculate lrmsd for each vertex
    
    lrmsd = np.zeros(NV, np.float64)
    
    for vi in range(NV):
        
        # construct list of neighbors
        IXvi = np.floor((pos[DIM*vi:DIM*vi+DIM] - Lmin) / L * Ncells).astype(np.int32)
        neighbor_verts = [np.int32(x) for x in range(0)]
        for ix in [-1, 0, 1]:
            IX = (IXvi[0] + ix + Ncells[0]) % Ncells[0]
            for iy in [-1, 0, 1]:
                IY = (IXvi[1] + iy + Ncells[1]) % Ncells[1]
                for iz in [-1, 0, 1]:
                    IZ = (IXvi[2] + iz + Ncells[2]) % Ncells[2]
                    
                    neighbor_verts.extend(neighbor_grid[IX][IY][IZ])

        # calc weights
        if weighted:
            weights = np.zeros(len(neighbor_verts), np.float64)
            
            for j in range(len(neighbor_verts)):
                vj = neighbor_verts[j]
                
                d = la.norm(pos[DIM*vj:DIM*vj+DIM] - pos[DIM*vi:DIM*vi+DIM])
                if d <= max_dist:
                    weights[j] = np.exp(-d**2 / (max_dist/3)**2 / 2)
            
        else:
            weights = None
                    
        # calc lrmsd
        lrmsd[vi] = calc_rmsd(vi, neighbor_verts, pos, disp, linear=linear, weighted=weighted, weights=weights)
        
    return lrmsd
        
                    
@njit
def calc_hinge_overlap(sectors_to_verts, pos, disp, linear=True):
    
    DIM = 3
    NV = len(pos)//DIM
    
    disp_hinge = np.zeros(DIM*NV, np.float64)
    for si in range(len(sectors_to_verts)):
        if len(sectors_to_verts[si]) > DIM:
            xcm, ucm, F = calc_global_motion(List(sectors_to_verts[si]), pos, disp)
            R, U = decompose_def_grad(F, linear=linear)
            for vi in sectors_to_verts[si]:
                disp_hinge[DIM*vi:DIM*vi+DIM] = (R - np.identity(DIM)) @ (pos[DIM*vi:DIM*vi+DIM] - xcm) + ucm;

    return disp_hinge.dot(disp) / la.norm(disp_hinge) / la.norm(disp) 