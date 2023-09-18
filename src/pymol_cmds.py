from pymol import cmd
from IPython.display import Image, display

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors


import itertools as it
import pickle


def set_display_props(prot_id):
    #     cmd.set("light_count", 10)
#     cmd.set("spec_count", 1)
#     cmd.set("shininess", 100)
    cmd.set("specular", 0.15)
#     cmd.set("ambient", 0)
#     cmd.set("direct", 0)
#     cmd.set("reflect", 1.5)
#     cmd.set("ray_shadow_decay_factor", 0.1)
#     cmd.set("ray_shadow_decay_range", 2)
    cmd.set("ray_shadow", "off")
     
    #set background
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", "on")
    
    if prot_id == 'GCK_b':
        
        cmd.set_view ((
    0.767848432,   -0.003914100,   -0.640618026,\
     0.109065734,   -0.984583557,    0.136742294,\
    -0.631278276,   -0.174867243,   -0.755584240,\
     0.000000000,   -0.000000000, -226.851272583,\
    -0.422599792,    0.466396332,   -0.475135803,\
   178.851272583,  274.851257324,  -20.000000000))
        
    elif prot_id == 'pyrB_pyrI_a':
        cmd.orient()
        
        cmd.set_view ((
     0.126922071,   -0.105888218,   -0.986241758,\
    -0.991905868,   -0.010979695,   -0.126471117,\
     0.002563915,    0.994316101,   -0.106425375,\
     0.000000000,    0.000000000, -482.058959961,\
     0.000007629,   -0.000099182,   -0.000930786,\
   380.058959961,  584.058959961,  -20.000000000))
    
    elif prot_id == 'HBA1_HBB_b':
        cmd.set_view ((
    0.434608489,    0.492343366,   -0.754130244,\
    -0.455213130,   -0.602419794,   -0.655641317,\
    -0.777106166,    0.628238618,   -0.037696462,\
     0.000000000,    0.000000000, -198.494873047,\
    -0.000427246,   -0.000423431,   -0.000011444,\
   156.494873047,  240.494873047,  -20.000000000))
        
    elif prot_id == 'Gch1_Gchfr':
        cmd.set_view ((
    -0.871213078,    0.382560015,   -0.307610661,\
     0.062842019,    0.708391130,    0.703011632,\
     0.486855656,    0.593147278,   -0.641203642,\
     0.000000000,    0.000000000, -425.346160889,\
    -0.000396729,    0.000000000,    0.000427246,\
   335.346160889,  515.346191406,  -20.000000000))
        
    elif prot_id == 'serA':
        cmd.set_view ((
    0.911044121,   -0.411946297,   -0.017268950,\
     0.412113726,    0.908530176,    0.068811916,\
    -0.012657464,   -0.069807492,    0.997480333,\
     0.000000000,    0.000000000, -396.989746094,\
    -0.000007629,    0.000007629,    0.000213623,\
   312.989746094,  480.989746094,  -20.000000000))
    
#     elif index == 48:
#         cmd.set_view ((
#     0.366661936,    0.542302668,    0.755954266,\
#     -0.929409266,    0.250122488,    0.271361560,\
#     -0.041921079,   -0.802088976,    0.595731556,\
#      0.000000000,    0.000000000, -255.207687378,\
#     -0.166919708,   -0.061561584,   -0.130485535,\
#    201.207687378,  309.207702637,  -20.000000000))
    
#     elif index == 3:
#         cmd.set_view ((
#      0.902963936,    0.252677202,   -0.347563148,\
#      0.210327297,   -0.965217531,   -0.155278578,\
#     -0.374709904,    0.067109033,   -0.924707174,\
#      0.000000000,    0.000000000, -109.899101257,\
#      0.000000000,    0.000000000,    0.000007629,\
#     61.899101257,  157.899108887,  -40.000000000))
       
#     elif index == 4:
#         cmd.set_view ((
#     -0.095419668,   -0.468310773,    0.878383636,\
#     -0.733796000,    0.629343331,    0.255825818,\
#     -0.672623694,   -0.620149672,   -0.403697908,\
#     -0.000000000,    0.000000000,  -96.161712646,\
#      0.206184387,    0.180732727,    0.112327576,\
#     54.161712646,  138.161712646,  -40.000000000))
        
#     elif index ==5:
#         cmd.set_view ((
#     -0.799379408,    0.498489022,    0.335403860,\
#      0.495720863,    0.231786206,    0.836976767,\
#      0.339483529,    0.835329950,   -0.432397962,\
#      0.000000000,    0.000000000, -123.636489868,\
#      0.031826019,    0.010704041,   -0.010997772,\
#     69.636489868,  177.636489868,  -40.000000000))
        
#     elif index == 6:
#         cmd.set_view ((
#     0.409856200,   -0.905033529,    0.113677442,\
#      0.022015406,   -0.114773989,   -0.993138254,\
#      0.911877275,    0.409549803,   -0.027116837,\
#      0.000000000,    0.000000000, -137.373870850,\
#     -0.000007629,   -0.000003815,    0.000003815,\
#     77.373870850,  197.373870850,  -40.000000000))
        
#     elif index == 7:
#         cmd.set_view ((
#     0.671964049,    0.677838802,   -0.298278183,\
#      0.322754323,    0.094453566,    0.941739619,\
#      0.666535378,   -0.729094923,   -0.155308515,\
#      0.000000000,    0.000000000, -164.848648071,\
#     -0.003551483,    0.010944366,   -0.002952576,\
#     92.848648071,  236.848648071,  -40.000000000))
        
#     elif index == 8:
#         cmd.set_view ((
#     -0.075487502,    0.037826896,   -0.996418834,\
#     -0.997045100,   -0.016872231,    0.074894182,\
#     -0.013980458,    0.999136031,    0.038988791,\
#     -0.000000000,    0.000000000, -164.848648071,\
#      0.000000000,    0.000007629,   -0.000553131,\
#     92.848648071,  236.848648071,  -40.000000000))
        
#     elif index == 9:
#         cmd.set_view ((
#     -0.106360242,   -0.850727856,    0.514726996,\
#     -0.525779128,    0.487488806,    0.697067499,\
#     -0.843940496,   -0.196493566,   -0.499145836,\
#     -0.000000000,    0.000000000, -164.848648071,\
#      0.033756256,   -0.012092590,    0.000411987,\
#     92.848648071,  236.848648071,  -40.000000000))
        
    else:
        
    #set camera properties
#     cmd.set("field_of_view", 40)
#     cmd.zoom()
#     cmd.zoom("center", 50)
        cmd.orient()

    


    
@cmd.extend
def show_disp(index):
    index = int(index)
        
    df_db = pd.read_excel('data/proteins.xlsx', sheet_name='allosteric')
    row = df_db.iloc[index]
    
    prot_id = row['protein_id']
    
    with open("data/" + prot_id + "/structure.pkl", 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        df_merged = data['merged structure']
        df_prot_def = data['deformed structure']
        
    cmd.load('data/' + prot_id + '/full_deformed.pdb')
    cmd.load('data/' + prot_id + '/merged_reference.pdb')
    cmd.load('data/' + prot_id + '/merged_deformed.pdb')
    
#     display(df_merged['lrmsd'])
    
    gray = '0xd9d9d9'
    cmd.color(gray)
    
    #import modevector script
    cmd.run("src/modevectors.py")
    
    scale = 0.7
    modevectors("merged_deformed", "merged_reference", headrgb="0.0,0.0,0.0", tailrgb="0.0,0.0,0.0", cutoff=2.0, 
                head=scale*1.0, tail=scale*0.3)
    
    magenta = '0x%02x%02x%02x' % (152,78,163)
    green = '0x%02x%02x%02x' % (0, 128, 0)
    cmd.set("sphere_scale", 1.0)
    cmd.set("sphere_transparency", 0.5)


# #     norm = mcolors.Normalize(vmin=0.0, vmax=df_merged['lrmsd'].quantile(0.95))
# #     cmaps = it.cycle([mpl.cm.Blues, mpl.cm.Reds, mpl.cm.Greens, mpl.cm.Purples, mpl.cm.Oranges, mpl.cm.Greys])
    
    
    # need to show any molecules in reference state
    # but also need to show both sets of binding sites
    
    # this should be inactive structure
    # but check merged structure for active/allo sites, skip overlap? check if doing this in calc
    for (chain_index, res_id, atom_name), row in df_prot_def.iterrows():
         
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
                    cmd.set("sphere_color", magenta, sele)
                    cmd.show("spheres", sele)
                elif active_site!=-1 and allo_site==-1:
                    cmd.set("sphere_color", green, sele)
                    cmd.show("spheres", sele)



    cmd.hide("everything", "merged_reference or merged_deformed")

    set_display_props(prot_id)
    
@cmd.extend
def show_topo(index):
    
    index = int(index)
        
    df_db = pd.read_excel('data/proteins.xlsx', sheet_name='allosteric')
    row = df_db.iloc[index]
    
    prot_id = row['protein_id']
    
    with open("data/" + prot_id + "/structure.pkl", 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        df_merged = data['merged structure']
        df_prot_ref = data['reference structure']

    lrmsd = df_merged['lrmsd'].values
    
    if row['n_sectors'] > 2:
        sector = df_merged['multi_sector'].values
    else:
        sector = df_merged['sector'].values

    cmd.load('data/' + prot_id + '/full_reference.pdb')
    
    
    magenta = '0x%02x%02x%02x' % (152,78,163)
    green = '0x%02x%02x%02x' % (0, 128, 0)
    cmd.set("sphere_scale", 1.0)
    cmd.set("sphere_transparency", 0.5)
    
    gray = '0xd9d9d9'
    cmd.color(gray)
    cmd.set("cartoon_transparency", 0.5)

#     yellow = '0x%02x%02x%02x' % (int(0.9254901960784314*256), int(0.8823529411764706*256), int(0.2*256))
    yellow = "gold"
    
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
               
            color = smaps[sector[row['merge_index']]].to_rgba(lrmsd[row['merge_index']], bytes=True)[0:3]
            color_hex = '0x%02x%02x%02x' % (color[0], color[1], color[2])
            cmd.color(color_hex, sele)
            cmd.set("cartoon_transparency", 0.0, sele)
            

    

    if 'allo_path' in df_merged.columns:

        df_path = df_merged.query("allo_path != -1").sort_values("allo_path")
        strain_path = []
        for (chain_index, res_id, atom_name), row in df_path.iterrows():

            chain_label = chr(ord('A') + chain_index)

            sele = "chain {} and resi {} and name {}".format(chain_label, res_id, atom_name) 
            strain_path.append(sele)

        for i in range(len(strain_path)-1):
            cmd.distance("allo_strain_path", strain_path[i], strain_path[i+1])

        cmd.set('dash_color', yellow, 'allo_strain_path')
        cmd.set('dash_radius', 0.7, 'allo_strain_path')
        

    elif 'coop_path' in df_merged.columns:
        df_path = df_merged.query("coop_path != -1").sort_values("coop_path")
        strain_path = []
        for (chain_index, res_id, atom_name), row in df_path.iterrows():

            chain_label = chr(ord('A') + chain_index)

            sele = "chain {} and resi {} and name {}".format(chain_label, res_id, atom_name) 
            strain_path.append(sele)

        for i in range(len(strain_path)-1):
            cmd.distance("coop_strain_path", strain_path[i], strain_path[i+1])

        cmd.set('dash_color', yellow, 'coop_strain_path')
        cmd.set('dash_radius', 0.7, 'coop_strain_path')

    
    cmd.hide("labels")
    cmd.set('dash_gap', 0.0)
    
    

    
    set_display_props(prot_id)
    
    
@cmd.extend
def show_paths(index):
    
    index = int(index)
        
    df_db = pd.read_excel('data/proteins.xlsx', sheet_name='allosteric')
    row = df_db.iloc[index]
    
    prot_id = row['protein_id']
    
    with open("data/" + prot_id + "/structure.pkl", 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        df_merged = data['merged structure']
        df_bonds_ref_merged = data['merged reference bonds']
        df_prot_ref = data['reference structure']

    lrmsd = df_merged['lrmsd'].values
    
    if row['n_sectors'] > 2:
        sector = df_merged['multi_sector'].values
    else:
        sector = df_merged['sector'].values

    cmd.load('data/' + prot_id + '/full_reference.pdb')
    
    
    magenta = '0x%02x%02x%02x' % (152,78,163)
    green = '0x%02x%02x%02x' % (0, 128, 0)
    cmd.set("sphere_scale", 1.0)
    cmd.set("sphere_transparency", 0.5)
    
    gray = '0xd9d9d9'
    cmd.color(gray)
    cmd.set("cartoon_transparency", 0.5)


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

#             color = smaps[sector[row['merge_index']]].to_rgba(lrmsd[row['merge_index']], bytes=True)[0:3]
#             color_hex = '0x%02x%02x%02x' % (color[0], color[1], color[2])
#             cmd.color(color_hex, sele)
#             cmd.set("cartoon_transparency", 0.0, sele)
            

    

    
    if 'allo_path_max_scale' in df_bonds_ref_merged.columns:

        norm = mcolors.Normalize(vmin=0.0, vmax=df_bonds_ref_merged['allo_path_max_scale'].max())
        for (chain_indexi, res_idi, atom_namei, chain_indexj, res_idj, atom_namej), row in df_bonds_ref_merged.iterrows():
            
            if row['allo_path_max_scale'] == 0.0:
                continue

            chain_labeli = chr(ord('A') + chain_indexi)
            chain_labelj = chr(ord('A') + chain_indexj)

            selei = "chain {} and resi {} and name {}".format(chain_labeli, res_idi, atom_namei) 
            selej = "chain {} and resi {} and name {}".format(chain_labelj, res_idj, atom_namej) 
            
            name = "{}_{}".format(row['edgei'], row['edgej'])
            cmd.distance(name, selei, selej)
            cmd.set('dash_color', 'gray', name)
            cmd.set('dash_radius', 0.7*norm(row['allo_path_max_scale']), name)
            
    elif 'coop_path_max_scale' in df_bonds_ref_merged.columns:

        norm = mcolors.Normalize(vmin=0.0, vmax=df_bonds_ref_merged['coop_path_max_scale'].max())
        for (chain_indexi, res_idi, atom_namei, chain_indexj, res_idj, atom_namej), row in df_bonds_ref_merged.iterrows():
            
            chain_labeli = chr(ord('A') + chain_indexi)
            chain_labelj = chr(ord('A') + chain_indexj)

            selei = "chain {} and resi {} and name {}".format(chain_labeli, res_idi, atom_namei) 
            selej = "chain {} and resi {} and name {}".format(chain_labelj, res_idj, atom_namej) 
            
            if row['coop_path_max_scale'] == 0.0:
                cmd.distance('edges', selei, selej)
                
            else:
            
                name = "{}_{}".format(row['edgei'], row['edgej'])
                cmd.distance(name, selei, selej)
                cmd.set('dash_color', 'blue', name)
                cmd.set('dash_radius', 0.05 + 0.7*norm(row['coop_path_max_scale']), name)
                
            
        cmd.set('dash_color', 'black', 'edges')
        cmd.set('dash_radius', 0.05, 'edges')


    if 'allo_path' in df_merged.columns:

        df_path = df_merged.query("allo_path != -1").sort_values("allo_path")
        strain_path = []
        for (chain_index, res_id, atom_name), row in df_path.iterrows():

            chain_label = chr(ord('A') + chain_index)

            sele = "chain {} and resi {} and name {}".format(chain_label, res_id, atom_name) 
            strain_path.append(sele)

        for i in range(len(strain_path)-1):
            cmd.distance("allo_strain_path", strain_path[i], strain_path[i+1])

        cmd.set('dash_color', '0x525252', 'allo_strain_path')
        cmd.set('dash_radius', 0.8, 'allo_strain_path')
        

    elif 'coop_path' in df_merged.columns:
        df_path = df_merged.query("coop_path != -1").sort_values("coop_path")
        strain_path = []
        for (chain_index, res_id, atom_name), row in df_path.iterrows():

            chain_label = chr(ord('A') + chain_index)

            sele = "chain {} and resi {} and name {}".format(chain_label, res_id, atom_name) 
            strain_path.append(sele)

        for i in range(len(strain_path)-1):
            cmd.distance("coop_strain_path", strain_path[i], strain_path[i+1])

        cmd.set('dash_color', '0x525252', 'coop_strain_path')
        cmd.set('dash_radius', 0.8, 'coop_strain_path')


    
    cmd.hide("labels")
    cmd.set('dash_gap', 0.0)
    
    cmd.hide("labels")
    cmd.set('dash_gap', 0.0)
    
    

    
    set_display_props(prot_id)
    
@cmd.extend
def show_domains(index):
    
    index = int(index)
        
    df_db = pd.read_excel('data/proteins.xlsx', sheet_name='allosteric')
    row = df_db.iloc[index]
    
    prot_id = row['protein_id']
    
    with open("data/" + prot_id + "/structure.pkl", 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        df_merged = data['merged structure']
        df_prot_ref = data['reference structure']

    lrmsd = df_merged['lrmsd'].values
    
    if 'multi_sector' in df_merged.columns:
        sector = df_merged['multi_sector'].values
    else:
        sector = df_merged['sector'].values

    cmd.load('data/' + prot_id + '/full_reference.pdb')
    
    
    magenta = '0x%02x%02x%02x' % (152,78,163)
    green = '0x%02x%02x%02x' % (0, 128, 0)
    cmd.set("sphere_scale", 1.0)
    cmd.set("sphere_transparency", 0.5)
    
    gray = '0xd9d9d9'
    cmd.color(gray)
    cmd.set("cartoon_transparency", 0.5)


    norm = mcolors.Normalize(vmin=0.0, vmax=df_merged['lrmsd'].quantile(0.95))
    cmaps = [mpl.cm.Blues, mpl.cm.Reds, mpl.cm.Greens, mpl.cm.Purples, mpl.cm.Oranges, mpl.cm.Greys]
    smaps = []
    for cmap in cmaps:
        cmap_cut = mcolors.LinearSegmentedColormap.from_list('cut', cmap(np.linspace(0.2, 0.9, cmap.N)))
        smaps.append(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cut))
#     smaps = [mpl.cm.ScalarMappable(norm=norm, cmap=cmap) for cmap in cmaps]
    
    colors = ['blue', 'red']

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
                    
                if 'allo_path' in df_merged.columns and df_merged.loc[idx, 'allo_path'] != -1:
                    
                    cmd.set("sphere_color", '0x525252', sele)
                    cmd.show("spheres", sele)
                    cmd.set("sphere_transparency", 0.0, sele)
                    
                    
#                 if 'coop_path' in df_merged.columns and df_merged.loc[idx, 'coop_path'] != -1:
                    
#                     cmd.set("sphere_color", '0x525252', sele)
#                     cmd.show("spheres", sele)
#                     cmd.set("sphere_transparency", 0.0, sele)
                    
            cmd.color(colors[sector[row['merge_index']]], sele)
            cmd.set("cartoon_transparency", 0.0, sele)
            

    

    if 'allo_path' in df_merged.columns:

        df_path = df_merged.query("allo_path != -1").sort_values("allo_path")
        strain_path = []
        for (chain_index, res_id, atom_name), row in df_path.iterrows():

            chain_label = chr(ord('A') + chain_index)

            sele = "chain {} and resi {} and name {}".format(chain_label, res_id, atom_name) 
            strain_path.append(sele)

        for i in range(len(strain_path)-1):
            cmd.distance("allo_strain_path", strain_path[i], strain_path[i+1])

        cmd.set('dash_color', '0x525252', 'allo_strain_path')
        cmd.set('dash_radius', 0.7, 'allo_strain_path')
    
    
#     if 'coop_path' in df_merged.columns:
#         df_path = df_merged.query("coop_path != -1").sort_values("coop_path")
#         strain_path = []
#         for (chain_index, res_id, atom_name), row in df_path.iterrows():

#             chain_label = chr(ord('A') + chain_index)

#             sele = "chain {} and resi {} and name {}".format(chain_label, res_id, atom_name) 
#             strain_path.append(sele)

#         for i in range(len(strain_path)-1):
#             cmd.distance("coop_strain_path", strain_path[i], strain_path[i+1])

#         cmd.set('dash_color', '0x525252', 'coop_strain_path')
#         cmd.set('dash_radius', 0.7, 'coop_strain_path')

    
    cmd.hide("labels")
    cmd.set('dash_gap', 0.0)
    
    

    
    set_display_props(prot_id)
    
    

@cmd.extend
def show_prot_old():
    
    index = 8
    
    df_db = pd.read_excel('data/proteins.xlsx', sheet_name='allosteric')

    prot = df_db.iloc[index]['Alias(es)']
    PDB_id = df_db.iloc[index]['Inactive PDB']
    
    print(PDB_id)

    with open("data/" + prot + "/df_prot_bonds.pkl", 'rb') as pkl_file:
        data = pickle.load(pkl_file)

        df_prot = data['df_prot']
        df_bonds = data['df_bonds']
        
        
    display(df_prot)
    print(df_prot.columns)
    
    with open('data/'+prot+'/'+PDB_id+'_chain_map.pkl', 'rb') as pkl_file:
        chain_map = pickle.load(pkl_file)


    print(chain_map)
    
    inv_chain_map = {chain_map[key]:key for key in chain_map}

#     cmd.reinitialize()

    print('data/' + prot + '/pdb' + PDB_id + '_clean_full.pdb')

    cmd.load('data/' + prot + '/pdb' + PDB_id + '_clean_full.pdb')

    norm = mcolors.Normalize(vmin=0.0, vmax=df_prot['lrmsd'].quantile(0.95))
    cmaps = it.cycle([mpl.cm.Blues, mpl.cm.Reds, mpl.cm.Greens, mpl.cm.Purples, mpl.cm.Oranges, mpl.cm.Greys])


    cmd.hide("everything", "sol")
    
    cmd.set_name('pdb' + PDB_id + '_clean_full', PDB_id)
    
#     cmd.set("cartoon_transparency", 1.0, PDB_id)
#     cmd.set("sphere_transparency", 1.0, PDB_id)
#     cmd.set("stick_transparency", 1.0, PDB_id)

    
    print(df_prot.groupby("chain_ref").size())

    cmd.hide("everything", PDB_id)
    
    magenta = '0x%02x%02x%02x' % (152,78,163)
    green = '0x%02x%02x%02x' % (0, 128, 0)
    cmd.set("sphere_scale", 1.0)
    cmd.set("sphere_transparency", 0.0)
    
    for sector, group in df_prot.groupby("sector"):
        print(len(group))
        
        smap = mpl.cm.ScalarMappable(norm=norm, cmap=next(cmaps))

        for idx, row in group.iterrows():
                        
            chain = row['chain_ref']
            chain_copy = row['chain_copy_ref']
            
            chain_full = inv_chain_map[(chain, chain_copy)]
            
            sele = "chain {} and resi {} and name {}".format(chain_full, idx[1], idx[2]) 
            
            if row['allo_site']:
                cmd.set("sphere_color", magenta, sele)
#                 cmd.set("sphere_transparency", 0.0, sele)
#                 cmd.set("sphere_scale", 0.5, sele)
                cmd.show("spheres", sele)
            elif row['active_site']:
                cmd.set("sphere_color", green, sele)
#                 cmd.set("sphere_transparency", 0.0, sele)
#                 cmd.set("sphere_scale", 0.5, sele)
                cmd.show("spheres", sele)
            
            else:

                color = smap.to_rgba(row['lrmsd'], bytes=True)[0:3]
                color_hex = '0x%02x%02x%02x' % (color[0], color[1], color[2])

                cmd.show("cartoon", sele)
                cmd.color(color_hex, sele)

    
    
    
    df_path = df_prot.query("allo_path != -1").sort_values("allo_path")
    strain_path = []
    for idx, row in df_path.iterrows():
        
        chain = row['chain_ref']
        chain_copy = row['chain_copy_ref']
        chain_full = inv_chain_map[(chain, chain_copy)]
        strain_path.append("chain {} and resi {} and name {}".format(chain_full, idx[1], idx[2]))
        
    for i in range(len(strain_path)-1):
        cmd.distance("allo_strain_path", strain_path[i], strain_path[i+1])
        
        
    df_path = df_prot.query("coop_path != -1").sort_values("coop_path")
    strain_path = []
    for idx, row in df_path.iterrows():
        
        chain = row['chain_ref']
        chain_copy = row['chain_copy_ref']
        chain_full = inv_chain_map[(chain, chain_copy)]
        strain_path.append("chain {} and resi {} and name {}".format(chain_full, idx[1], idx[2]))
        
    for i in range(len(strain_path)-1):
        cmd.distance("coop_strain_path", strain_path[i], strain_path[i+1])
        
        
    cmd.hide("labels")
#     cmd.hide("everything, het")

    cmd.set('dash_color', 'black', 'allo_strain_path')
    cmd.set('dash_radius', 0.7, 'allo_strain_path')
    
    
    cmd.set('dash_color', '0x525252', 'coop_strain_path')
    cmd.set('dash_radius', 0.7, 'coop_strain_path')
    
    cmd.set('dash_gap', 0.0)
#     cmd.set('dash_radius', 0.1, 'interactions')
#     cmd.set('dash_gap', 0.0, 'interactions')
          
#     df_bonds = df_bonds.query("PDB_id==@PDB_id").copy()
#     exclude_bond_types=[('proximal', '')]
#     for (bond_type, bond_subtype) in exclude_bond_types:
#         df_bonds = df_bonds.query("bond_type!=@bond_type or bond_subtype!=@bond_subtype").copy()
    
#     display(df_bonds)
    
#     display(df_bonds.groupby(['bond_type', 'bond_subtype']).size())
    
#     display(df_bonds.groupby(['chaini', 'chainj']).size())
    
#     for idx, row, in df_bonds.reset_index().iterrows():
        
# #         if idx > 100:
# #             break

#         if row['bond_type'] == 'covalent' and row['bond_subtype'] == '':
#             continue
        
#         chaini = row['chaini']
#         chain_copyi = row['chain_copyi']
#         resi = row['residuei']
#         atomi = row['atom_namei']
#         chain_fulli = inv_chain_map[(chaini, chain_copyi)]
        
        
#         chainj = row['chainj']
#         chain_copyj = row['chain_copyj']
#         resj = row['residuej']
#         atomj = row['atom_namej']
#         chain_fullj = inv_chain_map[(chainj, chain_copyj)]
        
        

#         chain_full = inv_chain_map[(chain, chain_copy)]
        
#         cmd.distance("interactions",  "chain {} and resi {} and name {}".format(chain_fulli, resi, atomi),  "chain {} and resi {} and name {}".format(chain_fullj, resj, atomj))
# #         cmd.set_bond("stick_color", 'black', "chain {} and resi {} and name {}".format(chain_fulli, resi, atomi),  "chain {} and resi {} and name {}".format(chain_fullj, resj, atomj))
        

#     with open("data/" + prot + "/strain_path.pkl", 'rb') as pkl_file:
#         data = pickle.load(pkl_file)
#         path_everts = data['path_everts']
        
#     for (vi, vj) in path_everts:
        
#         rowi = df_prot.iloc[vi]  
#         chaini = rowi['chain_def']
#         chain_copyi = rowi['chain_copy_def']
#         chain_fulli = inv_chain_map[(chaini, chain_copyi)]
        
        
#         rowj = df_prot.iloc[vj]
#         chainj = rowj['chain_def']
#         chain_copyj = rowj['chain_copy_def']
#         chain_fullj = inv_chain_map[(chainj, chain_copyj)]
        

#         cmd.distance("strain_path",  "chain {} and resi {} and name {}".format(chain_fulli, rowi.name[1], rowi.name[2]),
#                      "chain {} and resi {} and name {}".format(chain_fullj, rowj.name[1], rowj.name[2]))
        
# #         break

#     cmd.hide("labels")
# #     cmd.hide("everything, het")

#     cmd.set('dash_color', 'black', 'strain_path')
#     cmd.set('dash_radius', 0.7, 'strain_path')
#     cmd.set('dash_gap', 0.0)
# #     cmd.set('dash_radius', 0.1, 'interactions')
# #     cmd.set('dash_gap', 0.0, 'interactions')
    
    com = df_prot[['x_ref', 'y_ref', 'z_ref']].values.mean(axis=0)
    set_display_props(com)
    
    