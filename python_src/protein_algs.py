import sys, os

sys.path.insert(0, '../../py_scripts/')
from IPython.display import display

import numpy as np
import pandas as pd

import Bio.PDB as PDB

import mech_deform as deform
import topo_algs as topo


def merge_structures(df_ref, df_def):
    
    df_prot_merged = df_ref.query("sub_mol==-1 and reg_mol==-1").drop(columns=['chain_id', 'chain_copy', 'PDB_id', 'sub_mol', 'reg_mol']).merge(df_def.query("sub_mol==-1 and reg_mol==-1").drop(columns=['chain_id', 'chain_copy', 'PDB_id', 'res_name', 'element', 'vdw_rad', 'sub_mol', 'reg_mol']), suffixes=('_ref', '_def'), left_index=True, right_index=True, sort=True)
    
    # this choice is arbitrary if both have active or allo sites specified
    df_prot_merged['active_site'] = np.maximum(df_prot_merged['active_site_ref'], df_prot_merged['active_site_def'])
    df_prot_merged['allo_site'] = np.maximum(df_prot_merged['allo_site_ref'], df_prot_merged['allo_site_def'])
        
    df_prot_merged.drop(columns=['active_site_ref', 'active_site_def', 'allo_site_ref', 'allo_site_def'], inplace=True)
#     display(df_prot_merged)
    
    df_def['merge_index'] = -1
    df_def.loc[df_prot_merged.index, 'merge_index'] = np.arange(len(df_prot_merged.index))
    
    df_ref['merge_index'] = -1
    df_ref.loc[df_prot_merged.index, 'merge_index'] = np.arange(len(df_prot_merged.index))
    
    NV = len(df_prot_merged.index)
    DIM = 3
    
    x_ref = df_prot_merged[['x_ref', 'y_ref', 'z_ref']].values.flatten()
    x_def = df_prot_merged[['x_def', 'y_def', 'z_def']].values.flatten()
        
    xcm, ucm, F = deform.calc_global_motion(np.arange(NV), x_ref, x_def-x_ref)
    R, U = deform.decompose_def_grad(F, linear=False)
    
    
#     disp2 = x_def - x_ref
    
#     disp2 = deform.subtract_global_motion2(x_ref, disp2)
    
    x_ref = deform.subtract_global_motion(x_ref, xcm, np.zeros(DIM, np.float64), np.identity(DIM, np.float64))
    x_def = deform.subtract_global_motion(x_def, xcm, ucm, R)
    disp = x_def - x_ref
    

#     print(disp)
#     print(disp2)
    
    df_prot_merged[['x_ref', 'y_ref', 'z_ref']] = pd.DataFrame(x_ref.reshape((-1, DIM)), index=df_prot_merged.index)
    df_prot_merged[['x_def', 'y_def', 'z_def']] = pd.DataFrame(x_def.reshape((-1, DIM)), index=df_prot_merged.index)
    df_prot_merged[['u_x', 'u_y', 'u_z']] = pd.DataFrame(disp.reshape((-1, DIM)), index=df_prot_merged.index)
    
    
    x_ref = df_ref[['x', 'y', 'z']].values.flatten()
    x_def = df_def[['x', 'y', 'z']].values.flatten()
    
    x_ref = deform.subtract_global_motion(x_ref, xcm, np.zeros(DIM, np.float64), np.identity(DIM, np.float64))
    x_def = deform.subtract_global_motion(x_def, xcm, ucm, R) 
    
    df_ref[['x', 'y', 'z']] = pd.DataFrame(x_ref.reshape((-1, DIM)), index=df_ref.index)
    df_def[['x', 'y', 'z']] = pd.DataFrame(x_def.reshape((-1, DIM)), index=df_def.index)
            
    return df_prot_merged

def merge_bonds(df_prot, df_bonds, df_prot_merged):
    
    merge_indexi = df_prot.loc[df_bonds.reset_index().set_index(['chain_indexi', 'res_idi', 'atom_namei']).index.values, 'merge_index'].values
    merge_indexj = df_prot.loc[df_bonds.reset_index().set_index(['chain_indexj', 'res_idj', 'atom_namej']).index.values, 'merge_index'].values

    valid = (merge_indexi != -1) & (merge_indexj != -1)
    
#     edgei = merge_indexi[valid]
#     edgej = merge_indexj[valid]
    
    # make merged bonds have the same index structure as df_bonds, but just add merge_index and limit to relevant nodes
    df_bonds_merged = df_bonds[valid].copy()
    df_bonds_merged['edgei'] = np.array(merge_indexi[valid], np.int32)
    df_bonds_merged['edgej'] = np.array(merge_indexj[valid], np.int32)
    
                
    return df_bonds_merged
            
    


def calc_local_rmsd(df_merged, l0=15):
    
    disp = df_merged[['u_x', 'u_y', 'u_z']].values.flatten()
    
    x_ref = df_merged[['x_ref', 'y_ref', 'z_ref']].values.flatten()
    
    lrmsd = deform.calc_local_rmsd(np.array(x_ref, np.float64), np.array(disp, np.float64), l0, linear=False, weighted=True)
        
    df_merged['lrmsd'] = lrmsd

    

def df_to_pdb(prot_id, df_prot, suffix='', label=''):
    
    
    # create two structures, one reference and one deformed, 
    # as separate models in pdb file
    structure = PDB.Structure.Structure(label)
    model = PDB.Model.Model(0)
    structure.add(model)
    
    index = 1
    for chain_index, chain_group in df_prot.groupby('chain_index'):
        
        chain = PDB.Chain.Chain(chr(ord('A') + chain_index))
            
        model.add(chain)
        
        for res_id, res_group in chain_group.groupby('res_id'):
            
            if 'sub_mol' in res_group.columns and res_group['sub_mol'].values[0] != -1:
                entry_type = 'HET_S'
            elif 'reg_mol' in res_group.columns and res_group['reg_mol'].values[0] != -1:
                entry_type = 'HET_R'
            else:
                entry_type = ' '
            
            res = PDB.Residue.Residue((entry_type, res_id, ' '), res_group['res_name'].values[0], '')
            
            chain.add(res)
            
            for (_, _, atom_name), row in res_group.iterrows():
                
                pos = row[['x'+suffix, 'y'+suffix, 'z'+suffix]].values.flatten()
                atom = PDB.Atom.Atom(atom_name, pos, 1.0, 1.0, ' ', atom_name, index, element=row['element'])
                
                res.add(atom)
                
                index += 1
    
    # create pdb file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save("data/" + prot_id + "/{}.pdb".format(label))   
    
    
    



def find_allo_strain_path(df_merged, df_bonds_merged):
    
#     NV = len(df_merged.index)
    
#     source_sites = df_merged.reset_index().query("allo_site!=-1 and active_site==-1").index.values
#     target_sites = df_merged.reset_index().query("allo_site==-1 and active_site!=-1").index.values

    edgei = df_bonds_merged['edgei'].values
    edgej = df_bonds_merged['edgej'].values
    
    lrmsd = df_merged['lrmsd'].values

    source_sites = []
    for site_index, site in df_merged.reset_index().query("allo_site!=-1 and active_site==-1").groupby('allo_site'):
        source_sites.append(site.index.values)
        
    target_sites = []
    for site_index, site in df_merged.reset_index().query("allo_site==-1 and active_site!=-1").groupby('active_site'):
        target_sites.append(site.index.values)
        
    
    skeleton, boundary_edges = topo.find_skeleton(edgei, edgej, lrmsd, ascending=False)
    
    path_scales, path_lengths, strain_paths, max_edge_scale = topo.find_strain_paths(source_sites, target_sites, skeleton, edgei, edgej, lrmsd, coop=False)
    
    return path_scales, path_lengths, strain_paths, max_edge_scale
    

def find_coop_strain_path(df_merged, df_bonds_merged):
    
#     NV = len(df_merged.index)

    edgei = df_bonds_merged['edgei'].values
    edgej = df_bonds_merged['edgej'].values
    
    lrmsd = df_merged['lrmsd'].values

    target_sites = []
    for site_index, site in df_merged.reset_index().query("allo_site==-1 and active_site!=-1").groupby('active_site'):
        target_sites.append(site.index.values)
    
    
    skeleton, boundary_edges = topo.find_skeleton(edgei, edgej, lrmsd, ascending=False)
    
    path_scales, path_lengths, strain_paths, max_edge_scale = topo.find_strain_paths(target_sites, target_sites, skeleton, edgei, edgej, lrmsd, coop=True)
    
    return path_scales, path_lengths, strain_paths, max_edge_scale
    
    
    

def find_hinge(df_merged, df_bonds_merged, N_sectors=2, min_size=None):
    
    edgei = df_bonds_merged['edgei'].values
    edgej = df_bonds_merged['edgej'].values
    
    lrmsd = df_merged['lrmsd'].values
    
    x_ref = df_merged[['x_ref', 'y_ref', 'z_ref']].values.flatten()
    disp = df_merged[['u_x', 'u_y', 'u_z']].values.flatten().astype(np.float64)

    skeleton, boundary_edges = topo.find_skeleton(edgei, edgej, lrmsd)
    
    hinge_scale, hinge_overlap, sectors_to_verts, verts_to_sectors, sector_boundary_edges = topo.find_hinge(skeleton, boundary_edges, edgei, edgej, x_ref, disp, lrmsd, N_sectors=N_sectors, linear=False, min_size=min_size, maximize_overlap=False)
    
    return hinge_scale, hinge_overlap, sectors_to_verts


def find_hinge_sequence(df_merged, df_bonds_merged, N_sectors=2, min_size=None):
    
    
    edgei = df_bonds_merged['edgei'].values
    edgej = df_bonds_merged['edgej'].values
    
    lrmsd = df_merged['lrmsd'].values
    
    x_ref = df_merged[['x_ref', 'y_ref', 'z_ref']].values.flatten()
    disp = df_merged[['u_x', 'u_y', 'u_z']].values.flatten().astype(np.float64)

    skeleton, boundary_edges = topo.find_skeleton(edgei, edgej, lrmsd)
    
    hinge_scales = []
    hinge_overlaps = []
    for n in range(N_sectors):
        
        hinge_scale, hinge_overlap, sectors_to_verts, verts_to_sectors, sector_boundary_edges = topo.find_hinge(skeleton, boundary_edges, edgei, edgej, x_ref, disp, lrmsd, N_sectors=2, linear=False, min_size=min_size, maximize_overlap=False)
        
        hinge_scales.append(hinge_scale)
        hinge_overlaps.append(hinge_overlap)
        
        skeleton = list(set(skeleton) - set(sector_boundary_edges))
        boundary_edges = list(set(boundary_edges) - set(sector_boundary_edges))
        
        print(hinge_scales)
        print(hinge_overlaps)
        print([len(si) for si in sectors_to_verts])
        
    return hinge_scales, hinge_overlaps