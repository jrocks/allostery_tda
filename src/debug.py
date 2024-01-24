import sys, os

sys.path.insert(0, 'src/')

from pymol import cmd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

import pickle

import protein_algs as palgs
import topo_algs as topo

@cmd.extend
def show_debug(index):
    
    index = int(index)
        
    df_db = pd.read_excel('data/proteins.xlsx', sheet_name='allosteric')
    row = df_db.iloc[index]
    
    prot_id = row['protein_id']
    
    with open("data/" + prot_id + "/structure.pkl", 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        df_merged = data['merged structure']
        # df_prot_ref = data['reference structure']
        # df_bonds_ref_merged = data['merged reference bonds']
        df_prot_ref = data['deformed structure']
        df_bonds_ref_merged = data['merged reference bonds']

    cmd.load('data/' + prot_id + '/full_reference.pdb')
    
    
    magenta = '0x%02x%02x%02x' % (152,78,163)
    green = '0x%02x%02x%02x' % (0, 128, 0)
    cmd.set("sphere_scale", 1.0)
    cmd.set("sphere_transparency", 0.5)
    
    gray = '0xd9d9d9'
    cmd.color(gray)
    cmd.set("cartoon_transparency", 0.5)


    NV = len(df_merged.index)
    
    edgei = df_bonds_ref_merged['edgei'].values
    edgej = df_bonds_ref_merged['edgej'].values
    
    l0 = 15
    palgs.calc_local_rmsd(df_merged, l0=l0)
    
    lrmsd = df_merged['lrmsd'].values
    
    x_ref = df_merged[['x_ref', 'y_ref', 'z_ref']].values.flatten()
    disp = df_merged[['u_x', 'u_y', 'u_z']].values.flatten().astype(np.float64)
    
    

    skeleton, boundary_edges = topo.find_skeleton(edgei, edgej, lrmsd)
    
    sectors_to_verts, verts_to_sectors = topo.find_sectors(skeleton, NV, edgei, edgej)
    
    sector = np.zeros(len(lrmsd), "int64")
    for i, verts in enumerate(sectors_to_verts):
        sector[verts] = 0
    
    
    norm = mcolors.Normalize(vmin=0.0, vmax=df_merged['lrmsd'].quantile(0.95))
    cmaps = [mpl.cm.Blues, mpl.cm.Reds, mpl.cm.Greens, mpl.cm.Purples, mpl.cm.Oranges, mpl.cm.Greys]
    
    smaps = []
    for cmap in cmaps:
        cmap_cut = mcolors.LinearSegmentedColormap.from_list('cut', cmap(np.linspace(0.2, 0.9, cmap.N)))
        smaps.append(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cut))
#     smaps = [mpl.cm.ScalarMappable(norm=norm, cmap=cmap) for cmap in cmaps]

    for (chain_index, res_id, atom_name), row in df_prot_ref.iterrows():
          
        chain_label = chr(ord('A') + chain_index)
        sele = "chain {} and resi {} and name {}".format(chain_label, res_id, atom_name) 
            
        if row['reg_mol'] != -1:
            cmd.color(magenta, sele)
            cmd.set("cartoon_transparency", 0.0, sele)
        
        elif row['sub_mol'] != -1:
            cmd.color(green, sele)
            cmd.set("cartoon_transparency", 0.0, sele)
            
            
        elif row['merge_index'] != -1:
            idx = (chain_index, res_id, atom_name)
            if idx in df_merged.index:
                allo_site = df_merged.loc[idx, 'allo_site']
                active_site = df_merged.loc[idx, 'active_site']

                if allo_site!=-1 and active_site==-1:
                    
#                     CAsele = "chain {} and resi {} and name {}".format(chain_label, res_id, 'CA') 
                    
                    cmd.set("sphere_color", magenta, sele)
                    cmd.show("spheres", sele)
                elif active_site!=-1 and allo_site==-1:
                    
#                     CAsele = "chain {} and resi {} and name {}".format(chain_label, res_id, 'CA') 
                    
                    cmd.set("sphere_color", green, sele)
                    cmd.show("spheres", sele)
                    
                if 'allo_path' in df_merged.columns:
                    if df_merged.loc[idx, 'allo_path'] != -1:
                    
                        cmd.set("sphere_color", yellow, sele)
                        cmd.show("spheres", sele)
                        cmd.set("sphere_transparency", 0.0, sele)
                
                elif 'coop_path' in df_merged.columns:
                    if df_merged.loc[idx, 'coop_path'] != -1:
                    
                        cmd.set("sphere_color", yellow, sele)
                        cmd.show("spheres", sele)
                        cmd.set("sphere_transparency", 0.0, sele)
               
            color = smaps[sector[row['merge_index']]%6].to_rgba(lrmsd[row['merge_index']], bytes=True)[0:3]
            color_hex = '0x%02x%02x%02x' % (color[0], color[1], color[2])
            cmd.color(color_hex, sele)
            cmd.set("cartoon_transparency", 0.0, sele)
            
    
    cmd.hide("labels")
    cmd.set('dash_gap', 0.0)

    cmd.set("specular", 0.15)
    cmd.set("ray_shadow", "off")
     
    #set background
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", "on")
  
    cmd.orient()