import sys, os

from IPython.display import display


# sys.path.insert(0, '../../lib_persistent_homology/')
# sys.path.insert(0, '../../lib_persistent_homology/python_src/')

import numpy as np
import pandas as pd

import itertools as it
import pickle
import urllib.request
import json

import Bio.PDB as PDB
from openbabel import openbabel as ob

element_rad = {'N':1.55, 'C': 1.7, 'O':1.52, 'S': 1.8, 'F': 1.47,
               'FE': 1.25, 'CA': 2.31, 'ZN': 1.39, 'P': 1.8, 'H': 1.2,
               'BR': 1.85}

amino_code = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 
              'ASX': 'B', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 
              'GLX': 'Z', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 
              'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 
              'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 
              'TYR': 'Y', 'VAL': 'V'}


def preprocess(prot_id, PDB_id, check=False):
    """
    Description:

    Downloads and preprocesses pdb files. Preprocessing steps include:
    1. Download pdb file via BioPython.
    2. Clean pdb file using pdb-tools.
    3. Construct new pdb file of full x-ray structure. In this full structure, each chain copy is included explicitly as an additional chain with exactly one copy.
    4. Compute interactions using Arpeggio.

    Arguments:

    prot_id: Short name that will be used as protein identifier.
    PDB_id: PDB id of protein structure to be preprocessed.

    """
    
    PDB_id = PDB_id.lower()
    
    if check and os.path.exists('data/' + prot_id + '/' + PDB_id + '.json'):
        print("Preprocessing completed previously")
        return


    print("Preprocessing", prot_id, PDB_id)

    os.system('mkdir -p data/' + prot_id)

    PDB_file = "data/" + prot_id + "/" + PDB_id + ".pdb"
    mmCIF_file = "data/" + prot_id + "/" + PDB_id + ".cif"
    
    # Download pdb file
    if check and not os.path.exists(PDB_file):
        urllib.request.urlretrieve("https://files.rcsb.org/download/" + PDB_id + ".pdb1", PDB_file)
        urllib.request.urlretrieve("https://files.rcsb.org/download/" + PDB_id + ".cif", mmCIF_file)
    
    # load structure
    parser = PDB.PDBParser()
    structure = parser.get_structure(PDB_id, PDB_file)
        
    full_model = PDB.Model.Model(0)
    
    orig_to_full_chain_map = {}
    
    # combine all models (representing all pieces of full biological assembly) into single structure
    index = 0
    for i, model in enumerate(structure.get_models()):
        
        print(model.get_id())
        for chain in model.get_chains():
                        
            # remove residues with insertion code specifying alternative structures
            remove = []
            for res in chain.get_residues():
                (het_flag, seq_id, insert_code) = res.get_id()

                if insert_code != " ":
                    remove.append((het_flag, seq_id, insert_code))

            for res in remove:
                chain.detach_child(res)

            chain_copy = chain.copy()
            chain_copy.id = chr(ord('A') + index)

            orig_to_full_chain_map[(chain.id, i+1)] = chain_copy.id

            full_model.add(chain_copy)

            index += 1
            
    full_structure = PDB.Structure.Structure(PDB_id)
    full_structure.add(full_model)
    
    
    print("Original to full chain map:", orig_to_full_chain_map)

    
    # Save chain map
    with open('data/'+prot_id+'/chain_maps_'+PDB_id+'.pkl', 'wb') as pkl_file:
        full_to_orig_chain_map = {orig_to_full_chain_map[key]: key for key in orig_to_full_chain_map}
        data = {'full_to_orig_chain_map': full_to_orig_chain_map, 'orig_to_full_chain_map': orig_to_full_chain_map}
        pickle.dump(data, pkl_file)
        
        
     # Write structure back to mmCIF file
    io = PDB.mmcifio.MMCIFIO()
    io.set_structure(full_structure)
    io.save('data/' + prot_id + '/' + PDB_id + '.full.cif') 
       
    # get mmCIF header dictionary from original file
    mmcif_dict = PDB.MMCIF2Dict.MMCIF2Dict(mmCIF_file)
        
    # get mmCIF header dictionary from new mmCIF file
    full_mmcif_dict = PDB.MMCIF2Dict.MMCIF2Dict('data/' + prot_id + '/' + PDB_id + '.full.cif')
        
    # copy some parts of the header
    full_mmcif_dict['_chem_comp.id'] =  mmcif_dict['_chem_comp.id']
    full_mmcif_dict['_chem_comp.type' ] =  mmcif_dict['_chem_comp.type' ]
    full_mmcif_dict['_chem_comp.mon_nstd_flag' ] =  mmcif_dict['_chem_comp.mon_nstd_flag' ]
    full_mmcif_dict['_chem_comp.name' ] =  mmcif_dict['_chem_comp.name' ]
    full_mmcif_dict['_chem_comp.pdbx_synonyms' ] =  mmcif_dict['_chem_comp.pdbx_synonyms' ]
    full_mmcif_dict['_chem_comp.formula' ] =  mmcif_dict['_chem_comp.formula' ]
    full_mmcif_dict['_chem_comp.formula_weight'] =  mmcif_dict['_chem_comp.formula_weight'] 
    
     # Write structure back to mmCIF file again
    io.set_dict(full_mmcif_dict)
    io.save('data/' + prot_id + '/' + PDB_id + '.full.cif') 
    
    # Also write structure to pdb file
    io = PDB.PDBIO()
    io.set_structure(full_structure)
    io.save('data/' + prot_id + '/' + PDB_id + '.full.pdb') 
    
#     # Clean pdb file
#     # os.system('python ~/pdbtools/clean_pdb.py {}'.format('../data/' + prot_id + '/pdb' + PDB_id + '.full.pdb'))
    
#     # Check for special cases
#     check_special(PDB_id, 'data/' + prot_id + '/' + PDB_id + '.full.cif')
        
    print("Computing interactions...")
    # Compute intra-protein interactions    
    os.system('pdbe-arpeggio -sa data/{0}/{1}.full.cif -o data/{0}/'.format(prot_id, PDB_id))    

    print("Completed preprocessing...")

def check_special(PDB_id, PDB_file):
        
    # this pdb file has an extra digit in the thousands column for each residue
    # need to iterate through each row of clean pdb file, delete column, and replace zeros with white space
    if PDB_id == "1vg8":
        with open(PDB_file, 'r') as fn:
            lines = fn.readlines()
            
        for i in range(len(lines)):
            lines[i] = lines[i][:22] + str(int(lines[i][23:26])).rjust(4) + lines[i][26:]
           
        with open(PDB_file, 'w') as fn:
            for line in lines:
                fn.write(line)
    # this pdb file has uneccessary digits for chains, but necessary for HETATMS
    elif PDB_id=="1pj2" or PDB_id=="1qr6":
        with open(PDB_file, 'r') as fn:
            lines = fn.readlines()
            
        for i in range(len(lines)):
            if lines[i].startswith("ATOM"):
                lines[i] = lines[i][:22] + str(int(lines[i][23:26])).rjust(4) + lines[i][26:]
           
        with open(PDB_file, 'w') as fn:
            for line in lines:
                fn.write(line)    
        
    
def load_protein(prot_id, PDB_id, chain_list, reg_bind_list, sub_bind_list, exclude_bond_types=[]):
    
    print("Loading protein:", prot_id, PDB_id)
    
    PDB_id = PDB_id.lower()
    
    # global chain index
    chain_index = 0
    
    print("Loading atomic structure...")
    # load protein atomic structure
    df_list = []
    for chain in chain_list.split(","):
        split = chain.split(":")
        print(split)
        chain_id = split[0]
        chain_copy = int(split[1])
        res_name = None
        res_id = None
        atom_name = None
        
        if len(split) > 2:
            res_name = split[2]
            res_id = int(split[3])
        if len(split) > 4:
            atom_name = split[4]
                    
        # PDB, chain_id, chain_copy, res_name, res_id, atom_name
        selection = (PDB_id, chain_id, int(chain_copy), res_name, res_id, atom_name)
    
        df_sele = load_selection(prot_id, selection)
        df_sele['chain_index'] = chain_index
        chain_index += 1
            
        df_list.append(df_sele)
        
        
    df_prot = pd.concat(df_list)
     
    df_prot['active_site'] = -1
    df_prot['allo_site'] = -1
    df_prot['reg_mol'] = -1
    df_prot['sub_mol'] = -1
    
    
    print("Loading bonds...")
    # load protein bonds
    df_bonds = load_bonds(prot_id, PDB_id, exclude_bond_types=exclude_bond_types)
    
    df_list = []
    
    print("Loading regulatory molecules...")
    # load regulatory molecules and binding sites
    for sele in reg_bind_list.split(","):
        split = sele.split(":")
                
        # selections must always have pdb_id, chain_id and chain_copy specified
        
        # if PDB_id doesn't match, then skip
        if PDB_id != split[0].lower():
            continue        
        chain_id = split[1]
        chain_copy = int(split[2])
        
        query = "PDB_id==@PDB_id and chain_id==@chain_id and chain_copy==@chain_copy"
        
        # can also have optional res_name and res_id
        if len(split) >= 5:
            res_name = split[3]
            res_id = int(split[4])
            query += " and res_id==@res_id"
        else:
            res_name = None
            res_id = None
            
        # can also have optional atom name
        if len(split) == 6:
            atom_name = split[5]
            query += " and atom_name==@atom_name"
        else:
            atom_name = None
                    
        # if selection not part of protein
        df_query = df_prot.query(query)
        if len(df_query.index) == 0:
            df_sele = load_selection(prot_id, (PDB_id, chain_id, chain_copy, res_name, res_id, atom_name))
            # set chain_index to indicate regulatory molecule
            df_sele['chain_index'] = chain_index
            df_sele['active_site'] = -1
            df_sele['allo_site'] = -1
            df_sele['reg_mol'] = chain_index
            df_sele['sub_mol'] = -1
            chain_index += 1
            df_list.append(df_sele)
                        
            
        # if selection is a part of the protein 
        else:
            # mark as allosteric site
            df_prot.loc[df_query.index, 'allo_site'] = chain_index
            chain_index += 1
        
    
    # load substrate molecules and binding sites
    print("Loading substrate molecules...")
    for sele in sub_bind_list.split(","):
        split = sele.split(":")
                        
        # selections must always have pdb_id, chain_id and chain_copy specified
        
        # if PDB_id doesn't match, then skip
        if PDB_id != split[0].lower():
            continue        
        chain_id = split[1]
        chain_copy = int(split[2])
        
        query = "PDB_id==@PDB_id and chain_id==@chain_id and chain_copy==@chain_copy"
        
        # can also have optional res_name and res_id
        if len(split) >= 5:
            res_name = split[3]
            res_id = int(split[4])
            query += " and res_id==@res_id"
        else:
            res_name = None
            res_id = None
            
        # can also have optional atom name
        if len(split) == 6:
            atom_name = split[5]
            query += " and atom_name==@atom_name"
        else:
            atom_name = None
                    
        # if selection not part of protein
        df_query = df_prot.query(query)
        if len(df_query.index) == 0:
            df_sele = load_selection(prot_id, (PDB_id, chain_id, chain_copy, res_name, res_id, atom_name))
            # set chain_index to indicate substrate molecule
            df_sele['chain_index'] = chain_index
            df_sele['active_site'] = -1
            df_sele['allo_site'] = -1
            df_sele['reg_mol'] = -1
            df_sele['sub_mol'] = chain_index
            chain_index += 1
            df_list.append(df_sele)
        
        # if selection is a part of the protein 
        else:
            # mark as allosteric site
            df_prot.loc[df_query.index, 'active_site'] = chain_index
            chain_index += 1
    
    df_list.append(df_prot)
    
    df_prot = pd.concat(df_list)
    df_prot.set_index(['chain_id', 'chain_copy', 'res_id', 'atom_name'], inplace=True)
    
    print("Identifying allosteric and active sites...")
    # remove bonds between atoms that don't exist in df_prot
    # then add column for chain index for each atom to df_bonds
    df_bonds.reset_index(inplace=True)
    df_bonds.set_index(['chain_idi', 'chain_copyi', 'res_idi', 'atom_namei'], inplace=True)  
    df_bonds = df_bonds.loc[df_bonds.index.intersection(df_prot.index.values)].copy()
    df_bonds['chain_indexi'] = df_prot.loc[df_bonds.index.values, 'chain_index']
    
    df_bonds.reset_index(inplace=True)
    df_bonds.set_index(['chain_idj', 'chain_copyj', 'res_idj', 'atom_namej'], inplace=True)
    df_bonds = df_bonds.loc[df_bonds.index.intersection(df_prot.index.values)].copy()
    df_bonds['chain_indexj'] = df_prot.loc[df_bonds.index.values, 'chain_index']
    
    # set indices for both dataframes
    df_prot.reset_index(inplace=True)
    df_prot.set_index(['chain_index', 'res_id', 'atom_name'], inplace=True)
    df_bonds.reset_index(inplace=True)
    df_bonds.set_index(['chain_indexi', 'res_idi', 'atom_namei', 'chain_indexj', 'res_idj', 'atom_namej'], inplace=True)
    
    # identify binding sites on proteins by examining where regulatory and substrate molecules bind 
    # if multiple allosteric or active sites overlap, chooses arbitrary assignment
    for reg_mol, group in df_prot.query("reg_mol!=-1").groupby("reg_mol"):
                
        idx = set()
    
        df_bonds_tmp = df_bonds.reset_index().set_index(['chain_indexi', 'res_idi', 'atom_namei'])
        idx.update(df_bonds_tmp.loc[df_bonds_tmp.index.intersection(group.index.values)].reset_index().set_index(['chain_indexj', 'res_idj', 'atom_namej']).index.values)
    
        df_bonds_tmp = df_bonds.reset_index().set_index(['chain_indexj', 'res_idj', 'atom_namej'])
        idx.update(df_bonds_tmp.loc[df_bonds_tmp.index.intersection(group.index.values)].reset_index().set_index(['chain_indexi', 'res_idi', 'atom_namei']).index.values)
            
        idx = df_prot.query("reg_mol==-1 and sub_mol==-1").index.intersection(idx)
        df_prot.loc[idx, 'allo_site'] = reg_mol
    
    print("Allosteric Sites:")
    print(reg_bind_list)
    display(df_prot.query("reg_mol==-1 and sub_mol==-1 and allo_site!=-1").groupby('allo_site').size())
    
    
    for sub_mol, group in df_prot.query("sub_mol!=-1").groupby("sub_mol"):
        
        idx = set()
    
        df_bonds_tmp = df_bonds.reset_index().set_index(['chain_indexi', 'res_idi', 'atom_namei'])
        idx.update(df_bonds_tmp.loc[df_bonds_tmp.index.intersection(group.index.values)].reset_index().set_index(['chain_indexj', 'res_idj', 'atom_namej']).index.values)
    
        df_bonds_tmp = df_bonds.reset_index().set_index(['chain_indexj', 'res_idj', 'atom_namej'])
        idx.update(df_bonds_tmp.loc[df_bonds_tmp.index.intersection(group.index.values)].reset_index().set_index(['chain_indexi', 'res_idi', 'atom_namei']).index.values)
            
        idx = df_prot.query("reg_mol==-1 and sub_mol==-1").index.intersection(idx)
        df_prot.loc[idx, 'active_site'] = sub_mol
        
    print("Active Sites:")
    print(sub_bind_list)
    display(df_prot.query("reg_mol==-1 and sub_mol==-1 and active_site!=-1").groupby('active_site').size())
    
    

    
#     print("Protein Dataframe:")
#     display(df_prot)
#     print("Bond Dataframe:")
#     display(df_bonds)
#     print("Bond Types:")
#     display(df_bonds.groupby(['bond_type', 'bond_subtype']).size())
    
    return df_prot, df_bonds
    
    

    
# returns dataframe with selection indicated in terms of original chain_id and chain_copy 
def load_selection(prot_id, selection):
       
    (PDB_id, chain_id, chain_copy, res_name, res_id, atom_name) = selection
        
    parser = PDB.MMCIFParser()
    structure = parser.get_structure(PDB_id, 'data/' + prot_id + '/' + PDB_id + '.full.cif')
        
    
    with open('data/' + prot_id + '/chain_maps_' + PDB_id + '.pkl', 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        orig_to_full_chain_map = data['orig_to_full_chain_map']
                 
    full_chain_id = orig_to_full_chain_map[(chain_id, chain_copy)]
            
    s_list = []
    for chain in structure.get_chains():
                
        if chain.get_id() != full_chain_id:
            continue
        
        for res in chain.get_residues():
            
            (het_flag, seq_id, insert_code) = res.get_id()
            
            seq_id = int(seq_id)

            # Skip HET atoms by default unless res_name and res_id are explicitly specified
            if res_name is None and not het_flag.isspace():
                continue
            # Skip if residue is specified, but res_id does not match sequence id
            elif res_id is not None and seq_id != res_id:
                continue
#             # Skip is contains insertion code indicating alternative structure
#             elif insert_code != ' ':
#                 continue
                                    
            for atom in res.get_atoms():
                
                # Skip if atom name is specified, but doesn't match
                if atom_name is not None and atom.get_name() != atom_name:
                    continue
                
                s_list.append([PDB_id, chain_id, chain_copy, res.get_resname(), seq_id, atom.get_name(), 
                               atom.element, *atom.get_coord(), atom.get_bfactor()])   
                
    df = pd.DataFrame(s_list, columns=['PDB_id', 'chain_id', 'chain_copy', 'res_name', 'res_id', 'atom_name', 'element', 'x', 'y', 'z', 'bfactor'])

    df['vdw_rad'] = df['element'].map(element_rad) 
    
    
    return df 
    



def load_bonds(prot_id, PDB_id, exclude_bond_types=[]):
    
    
    # load pdb file to mad atom indices to anom names
    mmCIF_file = 'data/' + prot_id + '/' + PDB_id + '.full.cif'
    
    mmcif_dict = PDB.MMCIF2Dict.MMCIF2Dict(mmCIF_file)
        
    df_atom_map = pd.DataFrame({'ATOM': mmcif_dict['_atom_site.group_PDB'], 
                               'atom_index': mmcif_dict['_atom_site.id'],
                               'atom_name': mmcif_dict['_atom_site.label_atom_id']}).astype({'atom_index': 'int32'}).set_index('atom_index')
                
    df_atom_map = df_atom_map.query("ATOM=='ATOM' or ATOM=='HETATM'").copy()
                        
    # load chain map
    with open('data/' + prot_id + '/chain_maps_' + PDB_id + '.pkl', 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        full_to_orig_chain_map = data['full_to_orig_chain_map']
        
        
    print(full_to_orig_chain_map)
    
    PDB_file = 'data/' + prot_id + '/' + PDB_id + '.full.pdb'
    
    # load pdb file again using openbabel to find bonds based on standard chemistry of amino acids and polypeptides
    ob_conv = ob.OBConversion()
    ob_conv.SetInFormat('pdb')
    mol = ob.OBMol()
    ob_conv.ReadFile(mol, PDB_file)

    # iterate through bonds and add to list
    s_list = []
    for bond in ob.OBMolBondIter(mol):
        
        chaini = bond.GetBeginAtom().GetResidue().GetChain()
        chainj = bond.GetEndAtom().GetResidue().GetChain()
                
        res_idi = bond.GetBeginAtom().GetResidue().GetNum()
        res_idj = bond.GetEndAtom().GetResidue().GetNum()
                
        atom_namei = df_atom_map.loc[bond.GetBeginAtom().GetIdx(), 'atom_name']
        atom_namej = df_atom_map.loc[bond.GetEndAtom().GetIdx(), 'atom_name']
                
        tuplei = (*full_to_orig_chain_map[chaini], res_idi, atom_namei)
        tuplej = (*full_to_orig_chain_map[chainj], res_idj, atom_namej)
        
        # sort tuples in lexicographical order
        if tuplej < tuplei:
            tuplei, tuplej = tuplej, tuplei

        s_list.append([*tuplei, *tuplej, 'covalent', ''])
        
        
    # load contacts calculated by arpeggio
    
    #     Bond Types (mutually exclusive):
    #
    #     Clash 
    #         Denotes if the covalent radii of the two atoms are clashing, i.e. steric clash
    #     Covalent 
    #         Denotes if the two atoms appear to be covalently bonded
    #     VdW Clash 
    #         Denotes if the van der Waals radii of the two atoms are clashing
    #     VdW 
    #         Denotes if the van der Waals radii of the two atoms are interacting
    #     Proximal 
    #         Denotes the two atoms being > the VdW interaction distance, but with in 5 Angstroms of each other

    #     Bond Sub-types (can have more than one):
    #
    #     Hydrogen Bond 
    #         Denotes if the atoms form a hydrogen bond
    #     Weak Hydrogen Bond 
    #         Denotes if the atoms form a weak hydrogen bond
    #     Halogen Bond 
    #         Denotes if the atoms form a halogen bond
    #     Ionic 
    #         Denotes if the atoms may interact via charges
    #     Metal Complex 
    #         Denotes if the atoms are part of a metal complex
    #     Aromatic 
    #         Denotes two aromatic ring atoms interacting
    #     Hydrophobic 
    #         Denotes two hydrophobic atoms interacting
    #     Carbonyl 
    #         Denotes a carbonyl-carbon:carbonyl-carbon interaction
    #     Polar 
    #         Less strict hydrogen bonding (without angle terms)
    #     Weak Polar 
    #         Less strict weak hydrogen bonding (without angle terms)
    
    bond_types = {'clash', 'covalent', 'vdw_clash', 'vdw', 'proximal'}
    bond_subtypes = {'hbond', 'weak_hbond', 'xbond', 'ionic', 
                         'metal', 'aromatic', 'hydrophobic', 'carbonyl', 'polar', 'weak_polar'}
    
    contact_file = 'data/' + prot_id + '/' + PDB_id + '.json'
    
    with open(contact_file) as f:
        
        data = json.load(f)
        
        for entry in data:
            
            if entry['type'] != 'atom-atom':
                continue
                
            (chaini, res_idi, atom_namei) = (entry['bgn']['auth_asym_id'], entry['bgn']['auth_seq_id'], entry['bgn']['auth_atom_id'])
            (chainj, res_idj, atom_namej) = (entry['end']['auth_asym_id'], entry['end']['auth_seq_id'], entry['end']['auth_atom_id'])

            tuplei = (*full_to_orig_chain_map[chaini], int(res_idi), atom_namei)
            tuplej = (*full_to_orig_chain_map[chainj], int(res_idj), atom_namej)

            # sort tuples in lexicographical order
            if tuplej < tuplei:
                tuplei, tuplej = tuplej, tuplei
                
            
            bond_type = []
            bond_subtype = []
            
            for t in entry['contact']:
                
                if t in bond_types:
                    bond_type.append(t)
                    
                elif t in bond_subtypes:
                    bond_subtype.append(t)
        

            
            s_list.append([*tuplei, *tuplej, '/'.join(bond_type), '/'.join(bond_subtype)])
            
            
    
    df_bonds = pd.DataFrame(s_list, columns=['chain_idi', 'chain_copyi', 'res_idi', 'atom_namei', 
                                             'chain_idj', 'chain_copyj', 'res_idj', 'atom_namej', 'bond_type', 'bond_subtype'])
    
    
    for (bond_type, bond_subtype) in exclude_bond_types:
        if bond_subtype is None:
            df_bonds = df_bonds.query("bond_type!=@bond_type").copy()
        else:
            df_bonds = df_bonds.query("bond_type!=@bond_type or bond_subtype!=@bond_subtype").copy()
    
    
    df_bonds['PDB_id'] = PDB_id
        
    df_bonds.set_index(['PDB_id', 'chain_idi', 'chain_copyi', 'res_idi', 'atom_namei', 'chain_idj', 'chain_copyj', 'res_idj', 'atom_namej'], inplace=True)
    df_bonds.sort_index(inplace=True)
    
    df_bonds = df_bonds[~df_bonds.index.duplicated(keep='last')].copy()
    
    # display(df_bonds)
    
    return df_bonds
    