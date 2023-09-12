import numpy as np
import numpy.linalg as la

from numba import njit
from numba.typed import List

import mech_deform as deform


                
@njit
def construct_adj_list(edge_list, NV, edgei, edgej):
    
    adj_list = List([List([k for k in range(0)]) for vi in range(NV)])
    for ei in edge_list:
        vi = edgei[ei]
        vj = edgej[ei]
        adj_list[vi].append(vj)
        adj_list[vj].append(vi)
        
    return adj_list
        
        
@njit
def find_shortest_path(s, t, adj_list):
    
    NV = len(adj_list)
    
    dist = np.full(NV, -1, np.int32)
    prev = np.full(NV, -1, np.int32)
    visited = set()
    
    
    index = 0
    traversal = [s]
    dist[s] = 0
    
    while index < NV:
        
        curr = traversal[index]
        
        for vj in adj_list[curr]:
            if vj not in visited:
                dist[vj] = dist[curr] + 1
                prev[vj] = curr
                traversal.append(vj)
            
            if vj == t:
                path = [t]
                curr = t
                while curr != s:
                    curr = prev[curr]
                    path.append(curr)
                    
                # return path going from s to t
                return path[::-1]
                
                
        visited.add(curr)
        index += 1
        
        
@njit
def find_shortest_path_dists(vertices, adj_list):
    
    NV = len(adj_list)
    
    dist = np.full(NV, -1, np.int32)
    
    index = 0
    traversal = [np.int32(vi) for vi in vertices]
    for vi in vertices:
        dist[vi] = 0
    
    while index < len(traversal):
        
        curr = traversal[index]
        
        for vj in adj_list[curr]:
            if dist[vj] == -1 or dist[curr] + 1 < dist[vj]:
                dist[vj] = dist[curr] + 1
                traversal.append(vj)    
                
        index += 1
        
    return dist
        

@njit
def find_skeleton(edgei, edgej, F, ascending=True):
    

    NV = len(F)
    NE = len(edgei)
        
    digiF = np.argsort(np.argsort(F)).astype(np.int32)
        
    # create list of function values for edges
    # each function value is sorted tuple of function values for vertices (F[vi], F[vj]) where F[vi] > F[vj] is ascending
    # the edge index is also appended to each tuple in order to argsort within confines of numba implementation
    edge_F = []
    for ei in range(NE):
        Fi = digiF[edgei[ei]]
        Fj = digiF[edgej[ei]]
        
        if Fi < Fj and ascending or Fi > Fj and not ascending:
            edge_F.append((Fj, Fi, ei))
        else:
            edge_F.append((Fi, Fj, ei))
            
                        
    sorted_edges = [x[2] for x in sorted(edge_F, reverse=(not ascending))]
        
    # Map of sectors to sets of vertices contained within each sector
    # Sectors are indexed by edges
    sectors_to_verts = [{np.int32(-1)} for ei in range(NE)]
    # Reverse map of each vertex to the sector containing it
    # Default value of -1 indicates not assigned to sector
    verts_to_sectors = [-1 for vi in range(NV)]
    
    
    skeleton = []
    boundary_edges = []
    
    count = 0
    
    # Iterate through each edge in sorted order.
    for ei in sorted_edges:
        vi = edgei[ei]
        vj = edgej[ei]
        
        
        # Vertices are already contained within the same sector.
        # Nothing to do, skip
        if verts_to_sectors[vi] != -1 and verts_to_sectors[vj] != -1 and verts_to_sectors[vi] == verts_to_sectors[vj]:
            continue
        
        skeleton.append(ei)
        
        # Case I: Edge creates new connected component.
        # Neither vertex has been asigned to a new sector.
        # Create new sector and assigns both vertices to it.
        if verts_to_sectors[vi] == -1 and verts_to_sectors[vj] == -1:
            sectors_to_verts[ei] = {vi , vj}
            verts_to_sectors[vi] = ei
            verts_to_sectors[vj] = ei 
            
            count += 1
            
        # Case II: Edge merges two connected components.  
        # Vertices have previously been asigned to different sectors.
        # Merge sectors.
        elif verts_to_sectors[vi] != -1 and verts_to_sectors[vj] != -1 and verts_to_sectors[vi] != verts_to_sectors[vj]:
            si = verts_to_sectors[vi]
            sj = verts_to_sectors[vj]
            
            # order sectors so that si was created before sj
            if ascending and edge_F[sj] < edge_F[si] or not ascending and edge_F[si] < edge_F[sj]:
                si, sj = sj, si
                
            # Merge sector sj, the younger sector, to sector si, the older sector.
            sectors_to_verts[si] |= sectors_to_verts[sj]
            
            for vk in sectors_to_verts[sj]:
                 verts_to_sectors[vk] = si
            
            sectors_to_verts[sj] = {np.int32(-1)}
            
            boundary_edges.append(ei)
                                    
        # Case III: Edge increases size of one connected component.
        # One vertex has already belongs to a sector.
        # Add new vertex to the pre-exiting sector.
        else:
            if verts_to_sectors[vi] == -1:
                sj = verts_to_sectors[vj]
                sectors_to_verts[sj].add(vi)
                verts_to_sectors[vi] = sj
                
                # check if vi is a one vertex sector and add edge to boundary_verts if True
                if (ascending and digiF[vi] < digiF[vj]) or (not ascending and digiF[vj] < digiF[vi]):
                    boundary_edges.append(ei)
                
            else:
                si = verts_to_sectors[vi]
                sectors_to_verts[si].add(vj)
                verts_to_sectors[vj] = si
                
                # check if vj is a one vertex sector and add edge to boundary_verts if True
                if (ascending and digiF[vj] < digiF[vi]) or (not ascending and digiF[vi] < digiF[vj]):
                    boundary_edges.append(ei)
                
                
    return List(skeleton), List(boundary_edges)



@njit
def find_sectors(skeleton, NV, edgei, edgej):
    
    # Map of sectors to sets of vertices.
    # Each sector is initialized to contain exactly one vertex.
    sectors_to_verts = List([List([np.int32(vi)]) for vi in range(NV)])
    # Inverse map of vertices to sectors.
    verts_to_sectors = List([vi for vi in range(NV)])

    # Iterate through each edge.
    for ei in skeleton:
        vi = edgei[ei]
        vj = edgej[ei]

        si = verts_to_sectors[vi]
        sj = verts_to_sectors[vj]

        # Merge sector sj to sector si.
        for vk in sectors_to_verts[sj]:
            verts_to_sectors[vk] = si

        sectors_to_verts[si].extend(sectors_to_verts[sj])
        
        # empty list
        sectors_to_verts[sj] = List([np.int32(vk) for vk in range(0)])
                
    reduction_map = {}    
    reduced_sectors_to_verts = List()
    for si in range(NV):
        if len(sectors_to_verts[si]) > 0:
            reduction_map[si] = len(reduced_sectors_to_verts)
            reduced_sectors_to_verts.append(sectors_to_verts[si])
     
    reduced_verts_to_sectors = List()
    for vi in range(NV):
        reduced_verts_to_sectors.append(reduction_map[verts_to_sectors[vi]])
    
                
    return reduced_sectors_to_verts, reduced_verts_to_sectors


def find_strain_paths(source_sites, target_sites, skeleton, edgei, edgej, lrmsd, coop=False):
    
    NV = len(lrmsd)
    NE = len(edgei)
    
    adj_list = construct_adj_list(skeleton, NV, edgei, edgej)
    
    verts_to_edges = {tuple(sorted([edgei[i], edgej[i]])):i for i in range(NE)}
    
    max_edge_scale = np.zeros(NE, np.float64)
    
    strain_paths = []
    path_lengths = []
    path_scales = []
    for i, source in enumerate(source_sites):
        for j, target in enumerate(target_sites):
            if coop and i==j:
                continue
            
            path_candidates = []
            # find paths between each pair of nodes
            for inode in source:
                for onode in target:
                    path = find_shortest_path(inode, onode, adj_list)
                    path_candidates.append(path)
           
            # find most signicant path between sites
            lrmsd_min = []
            path_candidate_lengths = []
            for path in path_candidates:
                scale = lrmsd[path].min()
                lrmsd_min.append(scale)
                path_candidate_lengths.append(len(path))
                
                for k in range(len(path)-1):
                    ei = verts_to_edges[tuple(sorted([path[k], path[k+1]]))]
                    max_edge_scale[ei] = max(max_edge_scale[ei], scale)
                                        
            scale = np.max(lrmsd_min)
    
            # pick out all paths with max path scale
            imax_list = np.nonzero(scale==np.array(lrmsd_min))[0]
 
            # find path with minimum length
            imin = np.argmin(np.array(path_candidate_lengths)[imax_list])
        
            # record path
            strain_paths.append(path_candidates[imax_list[imin]])
            path_lengths.append(path_candidate_lengths[imax_list[imin]])
            path_scales.append(scale)
        
    # sort strain paths by decreasing path scales then increasing path lengths    
    asort = np.lexsort((path_lengths, -np.array(path_scales)))
    
    path_scales = np.array(path_scales)[asort]
    path_lengths = np.array(path_lengths)[asort]
    strain_paths = np.array(strain_paths)[asort]
        
    return path_scales, path_lengths, strain_paths, max_edge_scale

@njit
def find_hinge(skeleton, boundary_edges, edgei, edgej, pos, disp, lrmsd, N_sectors=2, linear=False, min_size=None, maximize_overlap=False):
    
    DIM = 3
    NV = len(pos)//DIM
    
    if min_size is None:
        min_size = DIM
    
    
    skeleton = set(skeleton)
    
    max_hinge_scale = -1.0
    max_hinge_overlap = 0.0
    
    # use greedy algorithm to choose boundary edges
    sector_boundary_edges = set()
    for n in range(N_sectors-1):

        # reset selection
        # only want to keep track of hinge scale for lest set of identified hinge sectors
        max_hinge_scale = -1.0
        max_hinge_overlap = 0.0
        selected_edge = -1
        
        # iterate through each boundary edge
        for i, bi in enumerate(boundary_edges):
            
            # skip edges that have already been selected as boundary edges
            if bi in sector_boundary_edges:
                continue
                
            reduced_skeleton = List(skeleton-{bi}-sector_boundary_edges)
            # remove boundary edge and identify sectors
            sectors_to_verts, verts_to_sectors = find_sectors(reduced_skeleton, NV, edgei, edgej)

            # calculate min sector size
            sector_sizes = [len(sectors_to_verts[si]) for si in range(len(sectors_to_verts))]
            min_sector_size = np.array(sector_sizes).min()


            # skip edges that create sectors smaller than the smallest allowed size
            if min_sector_size < min_size:
                continue
  
            if maximize_overlap:
                overlap = deform.calc_hinge_overlap(sectors_to_verts, pos, disp, linear=linear)
        
                if overlap > max_hinge_overlap:
                    max_hinge_overlap = overlap
                    selected_edge = bi
                    
                    
                    # calculate hinge scale
                    # calculate difference in deformation at hinge boundary from least uniform sector
                    boundary_max = np.max(np.array([lrmsd[edgei[bi]], lrmsd[edgej[bi]]]))

                    si = verts_to_sectors[edgei[bi]]
                    sj = verts_to_sectors[edgej[bi]]

                    si_min = np.min(lrmsd[np.array(list(sectors_to_verts[si]), np.int32)])
                    sj_min = np.min(lrmsd[np.array(list(sectors_to_verts[sj]), np.int32)])

                    hinge_scale = boundary_max - np.max(np.array([si_min, sj_min]))
                    
                    max_hinge_scale = hinge_scale

                    print(i, "/", len(boundary_edges))
                    print("Hinge Scale:", hinge_scale, "Overlap:", overlap)
                    print("Sector Sizes:", sector_sizes)
        
            else:
                                            
                # calculate hinge scale
                # calculate difference in deformation at hinge boundary from least uniform sector
                boundary_max = np.max(np.array([lrmsd[edgei[bi]], lrmsd[edgej[bi]]]))

                si = verts_to_sectors[edgei[bi]]
                sj = verts_to_sectors[edgej[bi]]

                si_min = np.min(lrmsd[np.array(list(sectors_to_verts[si]), np.int32)])
                sj_min = np.min(lrmsd[np.array(list(sectors_to_verts[sj]), np.int32)])

                hinge_scale = boundary_max - np.max(np.array([si_min, sj_min]))
            
                if hinge_scale > max_hinge_scale:
                    max_hinge_scale = hinge_scale
                    selected_edge = bi
                    
                    overlap = deform.calc_hinge_overlap(sectors_to_verts, pos, disp, linear=linear)
                
                    max_hinge_overlap = overlap
                    
                    
                    print(i, "/", len(boundary_edges))
                    print("Hinge Scale:", hinge_scale, "Overlap:", overlap)
                    print("Sector Sizes:", sector_sizes)
                    
            
        # add selected edge to list of boundary edges
        sector_boundary_edges.add(selected_edge)
                  
            
    if len(sector_boundary_edges) == 0:
        return 0.0, 0.0, None, None, None
        
        
    reduced_skeleton = List(skeleton-set(sector_boundary_edges))
    # remove boundary edge and identify sectors
    sectors_to_verts , verts_to_sectors = find_sectors(reduced_skeleton, NV, edgei, edgej)
    
    sector_sizes = [len(sectors_to_verts[si]) for si in range(len(sectors_to_verts))]
    
    print("Hinge Scale:", max_hinge_scale, "Overlap:", max_hinge_overlap)
    print("Sector Sizes:", sector_sizes)
        
    return max_hinge_scale, max_hinge_overlap, sectors_to_verts, verts_to_sectors, List(sector_boundary_edges)
    
 