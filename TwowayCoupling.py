# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:13:27 2023

@author: Chakr
"""


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import time
import os
from matplotlib import cm
from shapely.geometry import Polygon
import multiprocessing as mp
from queue import Empty
from queue import Full


def create_results_dir():
    '''
    Returns
    -------
    result_dir : TYPE, string
        DESCRIPTION. Name of Results directory.

    '''
    
    parent_dir = os.path.dirname(__file__)
    script_name = os.path.basename(__file__).split('.')[0]
    result_dir = f"Results - {script_name}"
    path = os.path.join(parent_dir,result_dir)
    try:        
        os.mkdir(path)
        print("Directory %s created" %result_dir)
        print("Path: " + parent_dir + "\\" + result_dir)
        
    except OSError as error:
        print(error)
        
    return result_dir


def generate_tissue(NP_layers=9,SE_layers=1,hs=0.55):
    '''
    
    Parameters
    ----------
    NP_layers : TYPE, int
        DESCRIPTION. Layers of the hexagon-shaped lattice (neural plate). layers = 0 --> 1 cell. The default is 9.
    SE_layers : TYPE, int
        DESCRIPTION. Layers of the hexagon-shaped lattice (surface ectoderm). The default is 2.
    hs : TYPE, float
        DESCRIPTION. Edge length of regular hexagonal cell. The default is 0.55.

    Returns
    -------
    CellGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Cell network.
    VertexGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Vertex network.
    Boundary : TYPE, list
        DESCRIPTION. List of boundary vertices starting from 0, going anticlockwise.
    NP_Boundary: TYPE, list
        DESCRIPTION. List of boundary vertices of the neural plate, starting at the bottom-left corner (lowest index) going anticlockwise.
    x_vc : TYPE, float
        DESCRIPTION. x-coordinate of the initial position of the centre of the hexagonal lattice.
    y_vc : TYPE, float
        DESCRIPTION. y-coordinate of the initial position of the centre of the hexagonal lattice.
    
    Returns a tuple of (dictionary, dictionary, list)
    
    '''
    
    # ------------------------------------------------------------------------
    #                       Creating the Cell Graph
    # ------------------------------------------------------------------------
    
    layers = NP_layers + SE_layers
    
    # No. of cells given by formula: C_n = 1 + 6 * Sum of natural numbers upto n   
    cells = 1 + 6*int((layers/2)*(layers+1)) 
    
    CellGraph = nx.path_graph(cells)
    
    cell_rows = [layers+1]
    for r in range(1,2*(layers+1)-1):
    
        # Disconnecting nodes to form rows
        CellGraph.remove_edge(sum(cell_rows)-1,sum(cell_rows))
        cell_rows.append(cell_rows[r-1]+1) if r <= layers else cell_rows.append(cell_rows[r-1]-1)
    
    # Connecting the rows 
    ls = 0    
    for r in range(len(cell_rows)):
            
        for c in range(cell_rows[r]):
    
            if r < int((len(cell_rows)-1)/2):
                CellGraph.add_edge(ls+c,ls+c+cell_rows[r])
                CellGraph.add_edge(ls+c,ls+c+cell_rows[r]+1)
    
            if r > int((len(cell_rows)-1)/2):
                CellGraph.add_edge(ls+c,ls+c-cell_rows[r-1])
                CellGraph.add_edge(ls+c,ls+c-cell_rows[r-1]+1)
                
            CellGraph.nodes[ls+c]['SE'] = 1 if (r < SE_layers) or (r > 2*layers-SE_layers) or ((c < SE_layers) or (c >= cell_rows[r]-SE_layers)) else 0
                
        ls = sum(cell_rows[0:r+1])
    
    # ------------------------------------------------------------------------
    #                       Creating the Vertex Graph
    # ------------------------------------------------------------------------
    
    vertices = 6*pow(layers+1,2)
    VertexGraph = nx.path_graph(vertices,create_using=nx.DiGraph)
    
    vert_rows = [(2*i+1) for i in cell_rows] # list comprehension
    vert_rows.insert(int((len(cell_rows)-1)/2),vert_rows[int((len(cell_rows)-1)/2)])
    
    for r in range(1,len(vert_rows)):
    
        # Disconnecting nodes to form rows
        VertexGraph.remove_edge(sum(vert_rows[0:r])-1,sum(vert_rows[0:r]))
    
    # Connecting the rows
    ls = 0
    
    for r in range(len(vert_rows)):
    
        for c in range(0,vert_rows[r],2):
    
            if r < int(len(vert_rows)/2)-1:
                VertexGraph.add_edge(ls+c,ls+c+vert_rows[r]+1)
                VertexGraph.add_edge(ls+c+vert_rows[r]+1,ls+c)
                if c != 0:
                    VertexGraph.add_edge(ls+c,ls+c-1)
                    VertexGraph.add_edge(ls+c-1,ls+c-2)
    
            elif r == int(len(vert_rows)/2)-1:
                VertexGraph.add_edge(ls+c,ls+c+vert_rows[r])
                if c != 0:
                    VertexGraph.add_edge(ls+c,ls+c-1)
                    VertexGraph.add_edge(ls+c-1,ls+c-2)
                
            elif r == int(len(vert_rows)/2):
                VertexGraph.add_edge(ls+c,ls+c-vert_rows[r-1])
                if c != 0:
                    VertexGraph.add_edge(ls+c,ls+c-1)
                    VertexGraph.add_edge(ls+c-1,ls+c-2)
    
            else:
                VertexGraph.add_edge(ls+c,ls+c-vert_rows[r-1]+1)
                VertexGraph.add_edge(ls+c-vert_rows[r-1]+1,ls+c)
                if c != 0:
                    VertexGraph.add_edge(ls+c,ls+c-1)
                    VertexGraph.add_edge(ls+c-1,ls+c-2)
    
        ls = sum(vert_rows[0:r+1])
    
    # ------------------------------------------------------------------------
    #                  Position attribute - Vertex Graph
    # ------------------------------------------------------------------------
    
    lc_row = len(cell_rows)
    lv_row = len(vert_rows)
    
    hlv_row = int(lv_row/2)
    
    rt3 = math.sqrt(3)
    hr2 = hs*rt3/2
    
    # Initial position of the centre of the hexagonal lattice  
    x_vc, y_vc = hs*rt3*(layers+1), hs*(3*(lc_row+1)/4+1)
    
    vx_t = sum(vert_rows[0:hlv_row])+int((vert_rows[hlv_row]-1)/2)    
    vx_b = sum(vert_rows[0:hlv_row-1])+int((vert_rows[hlv_row]-1)/2)
    
    for r in range(hlv_row):
    
        for c in range(int((vert_rows[hlv_row+r]+1)/2)):
    
            ind_tl = vx_t+sum(vert_rows[hlv_row:hlv_row+r])-r-c
            ind_tr = vx_t+sum(vert_rows[hlv_row:hlv_row+r])-r+c
            ind_bl = vx_b-sum(vert_rows[hlv_row:hlv_row+r])+r-c
            ind_br = vx_b-sum(vert_rows[hlv_row:hlv_row+r])+r+c
            if c == 0:
                VertexGraph.nodes[ind_tl]['cell_hex'] = []
                VertexGraph.nodes[ind_tl]['pos'] = (x_vc,y_vc+hs*0.75+r*hs*1.5+pow(-1,r)*hs*0.25)
                VertexGraph.nodes[ind_bl]['cell_hex'] = []
                VertexGraph.nodes[ind_bl]['pos'] = (x_vc,y_vc-hs*0.75-r*hs*1.5-pow(-1,r)*hs*0.25)
    
            else:
                VertexGraph.nodes[ind_tl]['cell_hex'] = []
                VertexGraph.nodes[ind_tl]['pos'] = (x_vc-c*hr2,y_vc+hs*0.75+r*hs*1.5+pow(-1,r+c)*hs*0.25)
                VertexGraph.nodes[ind_tr]['cell_hex'] = []
                VertexGraph.nodes[ind_tr]['pos'] = (x_vc+c*hr2,y_vc+hs*0.75+r*hs*1.5+pow(-1,r+c)*hs*0.25)
                VertexGraph.nodes[ind_bl]['cell_hex'] = []
                VertexGraph.nodes[ind_bl]['pos'] = (x_vc-c*hr2,y_vc-hs*0.75-r*hs*1.5-pow(-1,r+c)*hs*0.25)
                VertexGraph.nodes[ind_br]['cell_hex'] = []
                VertexGraph.nodes[ind_br]['pos'] = (x_vc+c*hr2,y_vc-hs*0.75-r*hs*1.5-pow(-1,r+c)*hs*0.25)
    
    # ------------------------------------------------------------------------
    #             Assigning vertices - Cell and Vertex Graphs
    # ------------------------------------------------------------------------
    
    ls = 0
    
    for r in range(lc_row):
    
        if r <= hlv_row-1:
            ls = sum(vert_rows[0:r])
    
        else:
            ls = sum(vert_rows[0:r]) + 1
    
        for c in range(cell_rows[r]):
    
            ind = sum(cell_rows[0:r]) + c
            v_i = ls + c*2 
            CellGraph.nodes[ind]['vert_hex'] = []
            CellGraph.nodes[ind]['vert_hex'].append(v_i)                      # [0]: lower left
            CellGraph.nodes[ind]['vert_hex'].append(v_i+1)                    # [1]: lower middle
            CellGraph.nodes[ind]['vert_hex'].append(v_i+2)                    # [2]: lower right
            VertexGraph.nodes[v_i]['cell_hex'].append(ind)
            VertexGraph.nodes[v_i+1]['cell_hex'].append(ind)
            VertexGraph.nodes[v_i+2]['cell_hex'].append(ind)
    
            if r < hlv_row-1:
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r]+3)   # [3]: upper right
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r]+2)   # [4]: upper middle
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r]+1)   # [5]: upper left
                VertexGraph.nodes[v_i+vert_rows[r]+3]['cell_hex'].append(ind)
                VertexGraph.nodes[v_i+vert_rows[r]+2]['cell_hex'].append(ind)
                VertexGraph.nodes[v_i+vert_rows[r]+1]['cell_hex'].append(ind)
    
            elif r == hlv_row-1:
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r]+2)   # [3]: upper right
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r]+1)   # [4]: upper middle
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r])     # [5]: upper left
                VertexGraph.nodes[v_i+vert_rows[r]+2]['cell_hex'].append(ind)
                VertexGraph.nodes[v_i+vert_rows[r]+1]['cell_hex'].append(ind)
                VertexGraph.nodes[v_i+vert_rows[r]]['cell_hex'].append(ind)
    
            else:
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r]+1)   # [3]: upper right
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r])     # [4]: upper middle
                CellGraph.nodes[ind]['vert_hex'].append(v_i+vert_rows[r]-1)   # [5]: upper left
                VertexGraph.nodes[v_i+vert_rows[r]+1]['cell_hex'].append(ind)
                VertexGraph.nodes[v_i+vert_rows[r]]['cell_hex'].append(ind)
                VertexGraph.nodes[v_i+vert_rows[r]-1]['cell_hex'].append(ind)
                
    # ------------------------------------------------------------------------
    #                      List of boundary vertices 
    # ------------------------------------------------------------------------
    
    Boundary = [0]
    loop_complete = 0
    
    while loop_complete == 0:
    
        g_ind = Boundary[-1]
        node_neighbours = [nn for nn in VertexGraph[g_ind]]
    
        for ng_ind in node_neighbours:
    
            if ng_ind in Boundary:
                continue
    
            else:
                if len(VertexGraph[ng_ind]) == 2:
                    Boundary.append(ng_ind)
                    break
    
                elif len(VertexGraph[ng_ind]) == 3 and len(node_neighbours) == 2:
                    Boundary.append(ng_ind)
                    break
    
                else:
                    continue
    
        if Boundary[-1] == g_ind:
            loop_complete = 1        
            
    np0 = -1
        
    for j in range(vertices):
        
        for cv in VertexGraph.nodes[j]['cell_hex']:
            
            if CellGraph.nodes[cv]['SE'] != 1:
                np0 = j
                break
                
        if np0 != -1:
            break
            
    NP_Boundary = []
    if np0 > 0:
        NP_Boundary.append(np0)
        
    if len(NP_Boundary) != 0:     
        loop_complete = 0
    
        while loop_complete == 0:
    
            g_ind = NP_Boundary[-1]
    
            for ng_ind in VertexGraph[g_ind]:
    
                SEcell = 0
    
                if ng_ind in NP_Boundary:
                    continue
    
                else:
    
                    for cv in VertexGraph.nodes[ng_ind]['cell_hex']:
    
                        if CellGraph.nodes[cv]['SE'] == 1:
                            SEcell += 1
    
                    if SEcell != 0 and SEcell < len(VertexGraph.nodes[ng_ind]['cell_hex']):
                        NP_Boundary.append(ng_ind)
                        break
    
            if NP_Boundary[-1] == g_ind:
                loop_complete = 1 
    
    else:
        NP_Boundary = Boundary
  
    return CellGraph, VertexGraph, Boundary, NP_Boundary, x_vc, y_vc


def set_sim_params(result_dir,cells,NP_cells,SE_cells,hs,k_se,simtime):
    '''
    
    Parameters
    ----------
    result_dir : TYPE, string
        DESCRIPTION. Name of Results directory.
    cells : TYPE, int
        DESCRIPTION. Number of cells.
    NP_cells: TYPE, int
        DESCRIPTION. Number of cells in the neural plate.
    SE_cells: TYPE, int
        DESCRIPTION. Number of cells in the surface ectoderm.
    hs : TYPE, float
        DESCRIPTION. Edge length of regular hexagonal cell.
    k_se: TYPE, float
        DESCRIPTION. Multiplicative factor for SE cell elasticities.
    simtime : TYPE, float
        DESCRIPTION. Total time over which solutions are being computed.
    
    Returns
    -------
    par_name : TYPE, list of float
        DESCRIPTION. Default parameter values.
    params_list : TYPE, list of list of float
        DESCRIPTION. Parameter values to be swept over.

    '''
    
    cf_ini_def = 0.1         # 0.1
    fpm_def = 0              # 0.016 # float(1/60)
    duty_def = 0.33          # 0.33  # value must be less than 1
    dt_def = 1               # 1
    Ca_def = 0.77*1e-3       # 3.08*1e-3
    Wa_def = 1               # 0.9
    An_ini_def = 0.785       # 0.785 
    Pn_ini_def = 3.3         # 3.3
    An_min_def = 0.015       # 0.015
    Pn_min_def = 0.456       # 0.456
    flash_amp_def = 0.6      # 0.6
    flash_time_def = 67      # 67
    testpar_def = 0          # 0
    
    par_name = [cf_ini_def,fpm_def,duty_def,dt_def,Ca_def,Wa_def,An_ini_def,Pn_ini_def,An_min_def,Pn_min_def,flash_amp_def,flash_time_def,testpar_def]
    
    with open(f'{result_dir}\\Default parameters.txt', 'w') as f:
        f.write(f'Cells: {cells} ({NP_cells} NP, {SE_cells} SE)  \n')
        f.write(f'Scaling factor: {hs} (side length of regular hexagon) \n')
        f.write(f'Multiplicative factor for SE cell elasticities: {k_se} \n')
        f.write(f'Simulation Time: {simtime} (100 a.u. = 1 min) \n\n')
        f.write(f'0 : Fraction of cells activated initially (cf_ini) = {cf_ini_def}\n')
        f.write(f'1 : Flash per min per cell (fpm) = {fpm_def}\n')
        f.write(f'2 : Duty ratio (duty) = {duty_def}\n')
        f.write(f'3 : Timestep (dt) = {dt_def}\n')
        f.write(f'4 : Natural Area shrinkage - Rate constant (Ca) = {Ca_def}\n')
        f.write(f'5 : Natural Area shrinkage - Threshold constant (Wa) = {Wa_def}\n')
        f.write(f'6 : Natural Area - Initial value (An_ini) = {An_ini_def}\n')
        f.write(f'7 : Natural Perimeter - Initial value (Pn_ini) = {Pn_ini_def}\n')
        f.write(f'8 : Natural Area - Minimum value (An_min) = {An_min_def}\n')
        f.write(f'9 : Natural Perimeter - Minimum value (Pn_min) = {Pn_min_def}\n')
        f.write(f'10 : Ca2+ flash amplitude (flash_amp) = {flash_amp_def}\n')
        f.write(f'11 : Ca2+ flash duration (flash_time) = {flash_time_def}\n')
        
    # Parameter lists
    #  Note:- to avoid repeated runs for default parameters, only one of them should be included in sweep
    
    cf_ini_sweep = []
    fpm_sweep = []
    duty_sweep = []
    dt_sweep = []
    Ca_sweep = []
    Wa_sweep = []
    An_ini_sweep = [] # do not set = 0, throws divided by 0 error
    Pn_ini_sweep = [] # do not set = 0, throws divided by 0 error
    An_min_sweep = []
    Pn_min_sweep = []
    flash_amp_sweep = [] 
    flash_time_sweep = []
    testpar_sweep = []
    
    params_list = [cf_ini_sweep,fpm_sweep,duty_sweep,dt_sweep,Ca_sweep,Wa_sweep,An_ini_sweep,Pn_ini_sweep,An_min_sweep,Pn_min_sweep,flash_amp_sweep,flash_time_sweep,testpar_sweep]
 
    return par_name, params_list


def initialize_cells(CellGraph,VertexGraph,Boundary,NP_Boundary,cells,k_se,cf_ini,fpm,duty,dt,Ca,Wa,An_ini,Pn_ini,An_min,Pn_min,flash_amp,flash_time,testpar):
    '''

    Parameters
    ----------
    CellGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Cell network.
    VertexGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Vertex network.
    Boundary : TYPE, list
        DESCRIPTION. List of boundary vertices starting from 0, going anticlockwise.
    NP_Boundary: TYPE, list
        DESCRIPTION. List of boundary vertices of the neural plate, starting at the bottom-left corner (lowest index) going anticlockwise.
    cells : TYPE, int
        DESCRIPTION. Number of cells.
    k_se: TYPE, float
        DESCRIPTION. Multiplicative factor for SE cell elasticities.
    cf_ini : TYPE, float
        DESCRIPTION. Fraction of cells activated initially.
    fpm : TYPE, float
        DESCRIPTION. Flash per minute per cell.
    duty : TYPE, float
        DESCRIPTION. Duty ratio of the Ca2+ flashes.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    Ca : TYPE, float
        DESCRIPTION. Rate constant for the area ratchet ODE.
    Wa : TYPE, float
        DESCRIPTION. Threshold constant for the area ratchet ODE.
    An_ini : TYPE, float
        DESCRIPTION. Initial value of natural area (for all cells).
    Pn_ini : TYPE, float
        DESCRIPTION. Initial value of natural perimeter (for all cells).
    An_min : TYPE, float
        DESCRIPTION. Minimum value limit for the cell natural area.
    Pn_min : TYPE, float
        DESCRIPTION. Minimum value limit for the cell natural perimeter.
    flash_amp : TYPE, float
        DESCRIPTION. Amplitude of a Ca2+ flash.
    flash_time : TYPE, float
        DESCRIPTION. Time duration of a Ca2+ flash.
    testpar : TYPE, float
        DESCRIPTION. A dummy variable which can be swept for.

    Returns
    -------
    None.

    '''
    
    flash_frms = int(flash_time/dt)
    
    for i in range(cells):
        
        # --------------------------------------------------------------------
        #     Initialise 'Ca_conc' attribute for each node of Cell Graph
        # --------------------------------------------------------------------
        
        CellGraph.nodes[i]['Ca_conc'] = np.random.choice(2,p=[1-cf_ini,cf_ini]) if CellGraph.nodes[i]['SE'] != 1 else 0
        if CellGraph.nodes[i]['Ca_conc'] == 1:
            CellGraph.nodes[i]['Ca_timer'] = flash_frms
        else:
            CellGraph.nodes[i]['Ca_timer'] = 0
        CellGraph.nodes[i]['Ca_amp'] = CellGraph.nodes[i]['Ca_conc']*flash_amp
        CellGraph.nodes[i]['Ca_off_time'] = 0
        
        # --------------------------------------------------------------------
        #            Initialise 'n_area', 'n_peri' for each cell
        # --------------------------------------------------------------------
        
        CellGraph.nodes[i]['n_area'] = An_ini
        CellGraph.nodes[i]['n_peri'] = Pn_ini
        
        # --------------------------------------------------------------------
        #                  Initialize any other attributes
        # --------------------------------------------------------------------
        
        CellGraph.nodes[i]['Ka']  = 0.03 if CellGraph.nodes[i]['SE'] != 1 else k_se*0.03
        CellGraph.nodes[i]['Kp']  = 0.02 if CellGraph.nodes[i]['SE'] != 1 else k_se*0.02
        CellGraph.nodes[i]['Ca']  = Ca if CellGraph.nodes[i]['SE'] != 1 else 0
        CellGraph.nodes[i]['Wa']  = Wa if CellGraph.nodes[i]['SE'] != 1 else 0
        CellGraph.nodes[i]['n_area_min'] = An_min
        CellGraph.nodes[i]['n_peri_min'] = Pn_min
        CellGraph.nodes[i]['Tn0'] = 0.01 if CellGraph.nodes[i]['SE'] != 1 else 0.01
        
        CellGraph.nodes[i]['flash_amp'] = 1.111*CellGraph.nodes[i]['Ka'] + 0.567
        
        CellGraph.nodes[i]['xi']  = 0
        
        cell_vert = CellGraph.nodes[i]['vert_hex']
        
        for ind in range(-1,5):
            
            Tn0 = CellGraph.nodes[i]['Tn0']
            xi  = CellGraph.nodes[i]['xi']

            VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn'] = Tn0 + xi
            if (len(Boundary) == len(NP_Boundary)) and (cell_vert[ind] in Boundary) and (cell_vert[ind+1] in Boundary):
                VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn'] = 0.5*VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn']
                VertexGraph.edges[cell_vert[ind+1],cell_vert[ind]]['Tn'] = VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn']


def activate_cells(CellGraph,VertexGraph,Boundary,NP_Boundary,cells,k,fpm,duty,dt,flash_amp,flash_time,testpar):
    '''

    Parameters
    ----------
    CellGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Cell network.
    VertexGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Vertex network.
    Boundary : TYPE, list
        DESCRIPTION. List of boundary vertices starting from 0, going anticlockwise.
    NP_Boundary: TYPE, list
        DESCRIPTION. List of boundary vertices of the neural plate, starting at the bottom-left corner (lowest index) going anticlockwise.
    cells : TYPE, int
        DESCRIPTION. Number of cells.
    k : TYPE, int
        DESCRIPTION. Solution index.    
    fpm : TYPE, float
        DESCRIPTION. Flash per minute per cell.
    duty : TYPE, float
        DESCRIPTION. Duty ratio of the Ca2+ flashes.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    flash_amp : TYPE, float
        DESCRIPTION. Amplitude of a Ca2+ flash.
    flash_time : TYPE, float
        DESCRIPTION. Time duration of a Ca2+ flash.
    testpar : TYPE, float
        DESCRIPTION. A dummy variable which can be swept for.    

    Returns
    -------
    None.

    '''
    
    T_60 = 6000
    duty = duty if duty <= 1 else 1
    max_ref_time = flash_time/duty
    flash_frms = int(flash_time/dt)
    maxref_frms = int(max_ref_time/dt)
    
    pc_form = (fpm*60)/((T_60/dt)-(fpm*60*flash_time/(dt*duty)))
    pc_base = 0
    if pc_form < 0:
        pc_base = 0
    elif pc_form > 1:
        pc_base = 1
    else:
        pc_base = pc_form
    
    Ka_val = 0.21
    
    k_xi = 4
    
    Hill_coeff = 10      
    Hill_max = 1
    delta_xi = 0
    
    for i in range(cells):
        
        cell_prob = pc_base
        cell_vert = CellGraph.nodes[i]['vert_hex']
        if CellGraph.nodes[i]['SE'] != 1:
            ext_ind = 0
            nl = CellGraph.nodes[i]['n_peri']/6
            
            for ind in range(-1,5):
                
                xPos_1 = VertexGraph.nodes[cell_vert[ind]]['pos'][0]
                yPos_1 = VertexGraph.nodes[cell_vert[ind]]['pos'][1]
                xPos_2 = VertexGraph.nodes[cell_vert[ind+1]]['pos'][0]
                yPos_2 = VertexGraph.nodes[cell_vert[ind+1]]['pos'][1]
                dist = np.sqrt(np.power((xPos_1-xPos_2),2)+np.power((yPos_1-yPos_2),2))
                ext_ind = ext_ind + 1 if ((dist-nl)/nl) > 0.1 else ext_ind
            
            # NOTE: (dist-nl)/nl is the Stretch index. The stretch indices of all the edges in a cell are used to calculate delta_cell_prob.
            
            SSCC_sens = 1+2*np.heaviside(CellGraph.nodes[i]['Ka']-Ka_val,1)

            delta_cell_prob = SSCC_sens*((0.005*dt)*ext_ind) # the dt is part of the formula to determine pc from fpm (see thesis)
            cell_prob = cell_prob + delta_cell_prob
                
        if (CellGraph.nodes[i]['Ca_timer'] == 0):
            if CellGraph.nodes[i]['Ca_conc'] == 1:
                if maxref_frms-flash_frms == 0:
                    CellGraph.nodes[i]['Ca_conc'] = np.random.choice(2,p=[1-cell_prob,cell_prob]) if CellGraph.nodes[i]['SE'] != 1 else 0
                    if CellGraph.nodes[i]['Ca_conc'] == 1:
                        CellGraph.nodes[i]['Ca_timer'] = flash_frms
                    else:
                        CellGraph.nodes[i]['Ca_off_time'] = k*dt
                else:
                    CellGraph.nodes[i]['Ca_conc'] = 0
                    CellGraph.nodes[i]['Ca_off_time'] = k*dt
                    CellGraph.nodes[i]['Ca_timer'] = maxref_frms-flash_frms   
            else:
                CellGraph.nodes[i]['Ca_conc'] = np.random.choice(2,p=[1-cell_prob,cell_prob]) if CellGraph.nodes[i]['SE'] != 1 else 0
                if (CellGraph.nodes[i]['Ca_conc'] == 1):
                    CellGraph.nodes[i]['Ca_timer'] = flash_frms
            
            CellGraph.nodes[i]['flash_amp'] = 1.111*CellGraph.nodes[i]['Ka'] + 0.567
            CellGraph.nodes[i]['Ca_amp'] = CellGraph.nodes[i]['Ca_conc']*CellGraph.nodes[i]['flash_amp']
        
        Hill_const = 0.01
        H_Amp = 1e-4*np.exp(7.68*CellGraph.nodes[i]['Ca_amp'])
        if CellGraph.nodes[i]['Ca_conc'] == 1:
            time_elapsed = (flash_frms-CellGraph.nodes[i]['Ca_timer'])*dt
            delta_xi = dt*H_Amp*k_xi*Hill_derivative(Hill_coeff,Hill_max,Hill_const,flash_time,time_elapsed)
        else:
            time_elapsed = k*dt - CellGraph.nodes[i]['Ca_off_time']
            delta_xi = 0 if time_elapsed < flash_frms*dt else -1*CellGraph.nodes[i]['xi']
        
        CellGraph.nodes[i]['xi'] = CellGraph.nodes[i]['xi'] + delta_xi
        if CellGraph.nodes[i]['xi'] > k_xi*CellGraph.nodes[i]['Tn0']:
            CellGraph.nodes[i]['xi'] = k_xi*CellGraph.nodes[i]['Tn0']
        elif CellGraph.nodes[i]['xi'] < 0:
            CellGraph.nodes[i]['xi'] = 0
            
        cell_vert = CellGraph.nodes[i]['vert_hex']
        
        for ind in range(-1,5):
            
            Tn0 = CellGraph.nodes[i]['Tn0']
            xi  = CellGraph.nodes[i]['xi']

            VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn'] = Tn0 + xi
            if (len(Boundary) == len(NP_Boundary)) and (cell_vert[ind] in Boundary) and (cell_vert[ind+1] in Boundary):
                VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn'] = 0.5*VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn']
                VertexGraph.edges[cell_vert[ind+1],cell_vert[ind]]['Tn'] = VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn']


def update_vertex_position(VertexGraph,CellGraph,j,dx,dy,dt):
    '''

    Parameters
    ----------
    VertexGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Vertex network.
    CellGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Cell network.
    j : TYPE, int
        DESCRIPTION. Vertex index.
    dx : TYPE, float
        DESCRIPTION. Spatial stepsize along x-coordinate.
    dy : TYPE, float
        DESCRIPTION. Spatial stepsize along y-coordinate.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.

    Returns
    -------
    (xPos_next,yPos_next) : TYPE, tuple
        DESCRIPTION. Position of vertex at next timestep.

    '''
    
    U1x = 0
    U1y = 0
    U2x = 0
    U2y = 0
    
    #Lmin = 0
    
    for u in range(2): # loop index for U1 or U2
    
        for r in range(2): # loop index for x or y coordinate
    
            Ka = []
            A = []
            An = []
            Kp = []
            P = []
            Pn = []
            T = []
            AdC = []
            Ln = []
    
            for cell_ind in VertexGraph.nodes[j]['cell_hex']:
                                
                X = []
                Y = []
                
                for vrtx_ind in CellGraph.nodes[cell_ind]['vert_hex']:
                    
                    if (vrtx_ind == j):
                        if (u == 0 and r == 0):
                            X.append(VertexGraph.nodes[vrtx_ind]['pos'][0]-0.5*dx)
                            Y.append(VertexGraph.nodes[vrtx_ind]['pos'][1])
                            
                        elif (u == 0 and r == 1):
                            X.append(VertexGraph.nodes[vrtx_ind]['pos'][0])
                            Y.append(VertexGraph.nodes[vrtx_ind]['pos'][1]-0.5*dy)
                            
                        elif (u == 1 and r == 0):
                            X.append(VertexGraph.nodes[vrtx_ind]['pos'][0]+0.5*dx)
                            Y.append(VertexGraph.nodes[vrtx_ind]['pos'][1])
                            
                        else:
                            X.append(VertexGraph.nodes[vrtx_ind]['pos'][0])
                            Y.append(VertexGraph.nodes[vrtx_ind]['pos'][1]+0.5*dy)       
                    else:
                        X.append(VertexGraph.nodes[vrtx_ind]['pos'][0])
                        Y.append(VertexGraph.nodes[vrtx_ind]['pos'][1]) 
                
                Ka.append(CellGraph.nodes[cell_ind]['Ka'])
                A.append(cell_area(X,Y))
                An.append(CellGraph.nodes[cell_ind]['n_area'])
                Kp.append(CellGraph.nodes[cell_ind]['Kp'])
                P.append(cell_peri(X,Y))
                Pn.append(CellGraph.nodes[cell_ind]['n_peri'])
                
            for ne_vert in VertexGraph[j]:
                            
                xPos_1 = 0
                yPos_1 = 0
                xPos_2 = VertexGraph.nodes[ne_vert]['pos'][0]
                yPos_2 = VertexGraph.nodes[ne_vert]['pos'][1]
                if (u == 0 and r == 0):
                    xPos_1 = VertexGraph.nodes[j]['pos'][0] - 0.5*dx 
                    yPos_1 = VertexGraph.nodes[j]['pos'][1]
                    
                elif (u == 0 and r == 1):
                    xPos_1 = VertexGraph.nodes[j]['pos'][0]  
                    yPos_1 = VertexGraph.nodes[j]['pos'][1] - 0.5*dy
                    
                elif (u == 1 and r == 0):
                    xPos_1 = VertexGraph.nodes[j]['pos'][0] + 0.5*dx 
                    yPos_1 = VertexGraph.nodes[j]['pos'][1]
                    
                else:
                    xPos_1 = VertexGraph.nodes[j]['pos'][0] 
                    yPos_1 = VertexGraph.nodes[j]['pos'][1] + 0.5*dy
                    
                dist = np.sqrt(np.power((xPos_1-xPos_2),2)+np.power((yPos_1-yPos_2),2))
                
                adh_coeff = 0
                for cell_ind in VertexGraph.nodes[j]['cell_hex']:
                    
                    if ne_vert in CellGraph.nodes[cell_ind]['vert_hex']:
                        Tn0 = CellGraph.nodes[cell_ind]['Tn0']
                        xi  = CellGraph.nodes[cell_ind]['xi']
                        # function coefficients will have to be chosen according to the max value of Tn0 and xi 
                        #  and upon the min value of perimeter (and area, since they are linked)
                        adh_coeff = adh_coeff + (Tn0+xi)
                
                tension = VertexGraph.edges[ne_vert,j]['Tn'] + VertexGraph.edges[j,ne_vert]['Tn']
                
                T.append(tension)
                AdC.append(adh_coeff)
                Ln.append(dist)
            
            #Lmin = min(Ln)
            
            if (u == 0 and r == 0):
                U1x = enfunc(Ka,A,An,Kp,P,Pn,T,AdC,Ln)
            elif (u == 0 and r == 1):
                U1y = enfunc(Ka,A,An,Kp,P,Pn,T,AdC,Ln)
            elif (u == 1 and r == 0):
                U2x = enfunc(Ka,A,An,Kp,P,Pn,T,AdC,Ln)
            else:
                U2y = enfunc(Ka,A,An,Kp,P,Pn,T,AdC,Ln)
                           
    xPos = VertexGraph.nodes[j]['pos'][0]
    yPos = VertexGraph.nodes[j]['pos'][1]
    
    # Damping coefficient formula
    
    mu = 0
    for cell_ind in VertexGraph.nodes[j]['cell_hex']:
        
        mu = mu + (-3.65 + (2.56/(CellGraph.nodes[cell_ind]['n_peri']/6)))
    
    xPos_next = fwd_euler(mu,dt,xPos,U2x-U1x,dx)
    yPos_next = fwd_euler(mu,dt,yPos,U2y-U1y,dy)
    
    return (xPos_next,yPos_next)


def compute_cell_shape(CellGraph,VertexGraph,i,dt,testpar):
    '''

    Parameters
    ----------
    CellGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Cell network.
    VertexGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Vertex network.
    i : TYPE, int
        DESCRIPTION. Cell index.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    testpar : TYPE, float
        DESCRIPTION. A dummy variable which can be swept for.    

    Returns
    -------
    area, peri, na_next, np_next, U_cell : TYPE, tuple of float
        DESCRIPTION. Area, perimeter, natural area and natural perimeter at the next timestep, and potential energy of the cell.

    '''
    
    X = []
    Y = []
    T = []
    
    cell_vert = CellGraph.nodes[i]['vert_hex']
    
    for ind in range(-1,5):
        
        X.append(VertexGraph.nodes[cell_vert[ind]]['pos'][0])
        Y.append(VertexGraph.nodes[cell_vert[ind]]['pos'][1])
        T.append(VertexGraph.edges[cell_vert[ind],cell_vert[ind+1]]['Tn'])
        
    area = cell_area(X,Y)
    peri = cell_peri(X,Y)
    na_pres = CellGraph.nodes[i]['n_area']
    np_pres = CellGraph.nodes[i]['n_peri']
    
    # ------------------------------------------------------------------------
    #                          Potential energy
    # ------------------------------------------------------------------------

    Ka = CellGraph.nodes[i]['Ka']
    Kp = CellGraph.nodes[i]['Kp']
    
    U_cell = 0
    
    # Elastic energy summation
    U_cell = U_cell + 0.5*Ka*na_pres*np.power((area/na_pres-1),2)
    U_cell = U_cell + 0.5*Kp*np_pres*np.power((peri/np_pres-1),2)
    
    # Line energy summation
    for ind in range(-1,5):
        U_cell = U_cell + T[ind]*np.sqrt(np.power((X[ind+1]-X[ind]),2) + np.power((Y[ind+1]-Y[ind]),2))
        
    # ------------------------------------------------------------------------
    #                          Ratchet mechanism
    # ------------------------------------------------------------------------
    
    Ca = CellGraph.nodes[i]['Ca']
    Wa = CellGraph.nodes[i]['Wa']
    An_min = CellGraph.nodes[i]['n_area_min']
    #Pn_min = CellGraph.nodes[i]['n_peri_min']
        
    r_Ka = 2.7*1e-4 if CellGraph.nodes[i]['SE'] != 1 else 0
    r_Kp = 1.8*1e-3 if CellGraph.nodes[i]['SE'] != 1 else 0
    r_Tn = 2*9*1e-4 if CellGraph.nodes[i]['SE'] != 1 else 0
    
    Tn = CellGraph.nodes[i]['Tn0']
    
    na_next = 0
    np_next = 0
    
    if CellGraph.nodes[i]['xi'] != 0 and CellGraph.nodes[i]['Ca_conc'] == 0:
        Ka = Ka + dt*r_Ka
        Kp = Kp + dt*r_Kp
        Tn = Tn + dt*r_Tn
        
    if ((area - Wa*na_pres) < 0) and CellGraph.nodes[i]['Ca_conc'] == 0:
        na_next = na_pres + dt*-Ca
        np_next = 6*np.sqrt((2*na_next)/(3*np.sqrt(3)))    
    else:
        na_next = na_pres
        np_next = 6*np.sqrt((2*na_next)/(3*np.sqrt(3)))
        
    if (Ka <= 0.3):
        CellGraph.nodes[i]['Ka'] = Ka
        
    if (Kp <= 0.2):
        CellGraph.nodes[i]['Kp'] = Kp
      
    if (Tn <= 0.1):
        CellGraph.nodes[i]['Tn0'] = Tn
        
    if (na_next < An_min):
        na_next = An_min
        np_next = 6*np.sqrt((2*na_next)/(3*np.sqrt(3)))

    return area, peri, na_next, np_next, U_cell
    

def simulate_VM(result_dir,directory,CellGraph,VertexGraph,Boundary,NP_Boundary,TissueCentre_x,TissueCentre_y,cells,vertices,simtime,hs,k_se,dx,dy,default_parameters,DEFAULT=True,par_ctr=None,par_ind=None,par=None,s_no=None,q_tstep=None,q_full=None,q_errp=None):
    '''

    Parameters
    ----------
    result_dir : TYPE, string
        DESCRIPTION. Name of Results directory.
    directory : TYPE, string
        DESCRIPTION. Name of Parameter directory.
    CellGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Cell network.
    VertexGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Vertex network.
    Boundary : TYPE, list
        DESCRIPTION. List of boundary vertices starting from 0, going anticlockwise.
    NP_Boundary: TYPE, list
        DESCRIPTION. List of boundary vertices of the neural plate, starting at the bottom-left corner (lowest index) going anticlockwise.
    TissueCentre_x : TYPE, float
        DESCRIPTION. x-coordinate of the initial position of the centre of the hexagonal lattice.
    TissueCentre_y : TYPE, float
        DESCRIPTION. y-coordinate of the initial position of the centre of the hexagonal lattice.
    cells : TYPE, int
        DESCRIPTION. Number of cells.
    vertices : TYPE, int
        DESCRIPTION. Number of vertices.
    simtime : TYPE, float
        DESCRIPTION. Total time over which solutions are being computed.
    hs : TYPE, float
        DESCRIPTION. Edge length of regular hexagonal cell.
    k_se: TYPE, float
        DESCRIPTION. Multiplicative factor for SE cell elasticities.
    dx : TYPE, float
        DESCRIPTION. Spatial stepsize along x-coordinate.
    dy : TYPE, float
        DESCRIPTION. Spatial stepsize along y-coordinate.
    default_parameters : TYPE, list of float
        DESCRIPTION. Default parameter values.
    DEFAULT : TYPE, boolean
        DESCRIPTION. Run over default parameters or perform parameter sweep. The default is True.
    par_ctr : TYPE, int
        DESCRIPTION. Process number.
    par_ind : TYPE, int
        DESCRIPTION. Parameter index.
    par : TYPE, float
        DESCRIPTION. Parameter value.
    s_no : TYPE, int
        DESCRIPTION. Sample number.
    q_tstep : TYPE, object of Queue class of multiprocessing package
        DESCRIPTION. Buffer queue holding progress percentage of each parallel process.
    q_full : TYPE, object of Value class of multiprocessing package
        DESCRIPTION. Flag reflecting full(1)/not full(0) status of q_tstep.
    q_errp : TYPE, object of Queue class of multiprocessing package
        DESCRIPTION. Buffer queue holding process identifiers of processes that have thrown errors.

    Returns
    -------
    err_flag : TYPE, int
        DESCRIPTION. Error flag when run is being performed for default parameters.

    '''
    
    err_flag = 0
    
    par_name = default_parameters
    if not DEFAULT:
        par_name[par_ind] = par
    cf_ini,fpm,duty,dt,Ca,Wa,An_ini,Pn_ini,An_min,Pn_min,flash_amp,flash_time,testpar = par_name
    
    try:
        
        with open(f'{result_dir}\\{directory}\\Cell_vertices.txt', 'w') as f:
            for i in range(cells):
                f.write(f'Cell {i}\n')
                for vh in CellGraph.nodes[i]['vert_hex']:
                    f.write(str(vh)+' ')   
                f.write('\n\n')
                
        with open(f'{result_dir}\\{directory}\\Vertex_cells.txt', 'w') as f:
            for j in range(vertices):
                f.write(f'Vertex {j}\n')
                for ch in VertexGraph.nodes[j]['cell_hex']:
                    f.write(str(ch)+' ')
                f.write('\n\n')
                
        with open(f'{result_dir}\\{directory}\\SE_cells.txt', 'w') as f:
            for i in range(cells):
                if CellGraph.nodes[i]['SE'] == 1:
                    f.write(str(i)+' ')
                            
        time_slns  = int(simtime/dt)
        # np.random.seed(0) # Optional: use this if you want the same Ca2+ activity in each run
        
        for k in range(0,time_slns):
            
            # ----------------------------------------------------
            #             Stage 1 - Cell activation
            # ----------------------------------------------------
            
            if k == 0:
                initialize_cells(CellGraph,VertexGraph,Boundary,NP_Boundary,cells,k_se,cf_ini,fpm,duty,dt,Ca,Wa,An_ini,Pn_ini,An_min,Pn_min,flash_amp,flash_time,testpar)
                
            else:
                activate_cells(CellGraph,VertexGraph,Boundary,NP_Boundary,cells,k,fpm,duty,dt,flash_amp,flash_time,testpar)
        
            # ----------------------------------------------------
            #             Stage 2 - Vertex updation
            # ----------------------------------------------------
    
            for j in range(vertices):
                
                if j in Boundary and (len(Boundary) != len(NP_Boundary)):
                    VertexGraph.nodes[j]['pos_next'] = VertexGraph.nodes[j]['pos']
                    continue
   
                VertexGraph.nodes[j]['pos_next'] = update_vertex_position(VertexGraph,CellGraph,j,dx,dy,dt)
                
            # ----------------------------------------------------
            #               Stage 3 - Cell updation
            # ----------------------------------------------------
            
            na_list    = []
            np_list    = []
            ucell_list = []
            
            for i in range(cells):
                
                area_buf, peri_buf, na_buf, np_buf, ucell_buff = compute_cell_shape(CellGraph,VertexGraph,i,dt,testpar)
                CellGraph.nodes[i]['area'] = area_buf
                CellGraph.nodes[i]['peri'] = peri_buf
                na_list.append(na_buf)
                np_list.append(np_buf)
                ucell_list.append(ucell_buff)
            
            # ----------------------------------------------------
            #              Stage 4 - Data extraction
            # ----------------------------------------------------
            
            # ----------------- To be computed -------------------
            
            # tot_area --> Area of NP
            # tot_peri --> Perimeter of NP
            # av_area  --> Average area of NP cells
            # av_peri  --> Average perimeter of NP cells
            # U_tot_np --> Potential energy of the NP
            # U_total  --> Potential energy of the NP + SE
            
            # --------------------- Unused -----------------------
            
            #area_celln     --> Area of Cell n
            #av_area_ncells --> Average area of n inner cells
 
            # ----------------------------------------------------
            
            vnum = len(NP_Boundary)
            X_tissue = []
            Y_tissue = []
            
            for bv in NP_Boundary:
                
                X_tissue.append(VertexGraph.nodes[bv]['pos'][0])
                Y_tissue.append(VertexGraph.nodes[bv]['pos'][1])
            
            tot_area = tissue_area(X_tissue,Y_tissue,vnum)
            tot_peri = 0
            
            for n in range(vnum):
                
                x1  = X_tissue[n-1]
                x2  = X_tissue[n]
                y1  = Y_tissue[n-1]
                y2  = Y_tissue[n]
                tot_peri = tot_peri + np.sqrt(np.power((x2-x1),2)+np.power((y2-y1),2))
            
            av_area = 0
            av_peri = 0
            U_tot_np = 0
            np_cells = 0
            
            for i in range(cells):
                
                if CellGraph.nodes[i]['SE'] != 1:
                    np_cells += 1
                    av_area = av_area + CellGraph.nodes[i]['area']
                    av_peri = av_peri + CellGraph.nodes[i]['peri']
                    U_tot_np = U_tot_np + ucell_list[i]
                    
            av_area = av_area/np_cells
            av_peri = av_peri/np_cells
            U_total = sum([ucell_list[i] for i in range(cells)])
            
            txtskip = int(time_slns/int(simtime/0.1)) if dt < 0.1 else 1 # limits size of data files for dt < 0.1

            if k % txtskip == 0:
                write_data(result_dir,directory,CellGraph,VertexGraph,cells,vertices,k,dt,tot_area,tot_peri,av_area,av_peri,U_tot_np,U_total)
            
            # ----------------------------------------------------
            #      Stage 5 - Update values for next timestep
            # ----------------------------------------------------
            
            for i in range(cells):
                
                CellGraph.nodes[i]['n_area'] = na_list[i]
                CellGraph.nodes[i]['n_peri'] = np_list[i]
                if (CellGraph.nodes[i]['Ca_timer'] > 0):
                    CellGraph.nodes[i]['Ca_timer'] -= 1
                    
            for j in range(vertices):
                
                VertexGraph.nodes[j]['pos'] = VertexGraph.nodes[j]['pos_next']
            
            if not DEFAULT:
                if q_full.value == 0:
                    try:
                        q_tstep.put((par_ctr,round((k/time_slns)*100,3)))
                    except Full:
                        clearQueue(q_tstep,q_full)
            
            else:
                print('Computation: ',k,' out of ',time_slns, 'solutions',end='\r')
                    
            # ----------------------------------------------------
            # ------------------ End of k loop -------------------
            # ----------------------------------------------------
        
        if not DEFAULT:
            if q_full.value == 0:
                try:
                    q_tstep.put((par_ctr,99.999))
                except Full:
                    clearQueue(q_tstep,q_full)
                    q_tstep.put((par_ctr,99.999))
            else:
                while q_full.value == 1:
                    continue
                q_tstep.put((par_ctr,99.999))
                
        else:
            print('                                                 ',end="")
            print('                                                 ',end='\r')
            print('Generating figures...')
            
        read_dir = f'{result_dir}\\{directory}'
        generate_figs(read_dir,cells,hs,TissueCentre_x,TissueCentre_y,dt,An_min,Pn_min,time_slns,txtskip)
        
        if not DEFAULT:
            if q_full.value == 0:
                try:
                    q_tstep.put((par_ctr,100.000))
                except Full:
                    clearQueue(q_tstep,q_full)
                    q_tstep.put((par_ctr,100.000))
            else:
                while q_full.value == 1:
                    continue
                q_tstep.put((par_ctr,100.000))
            q_tstep.close()
            
        else:
            print('Figures generated!\n')
    
    except (ValueError, IndexError, ZeroDivisionError, MemoryError, OverflowError, RuntimeError) as error:
        
        if not DEFAULT:
            q_errp.put((par_ind,par,s_no))
            if q_full.value == 0:
                try:
                    q_tstep.put((par_ctr,100.000))
                except Full:
                    clearQueue(q_tstep,q_full)
                    q_tstep.put((par_ctr,100.000))
            else:
                while q_full.value == 1:
                    continue
                q_tstep.put((par_ctr,100.000))
            q_tstep.close()
        
        else:
            print(error)
            err_flag = 1
            
    return err_flag
            

def write_data(result_dir,directory,CellGraph,VertexGraph,cells,vertices,k,dt,tot_area,tot_peri,av_area,av_peri,U_tot_np,U_total):
    '''

    Parameters
    ----------
    result_dir : TYPE, string
        DESCRIPTION. Name of Results directory.
    directory : TYPE, string
        DESCRIPTION. Name of Parameter directory.
    CellGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Cell network.
    VertexGraph : TYPE, dictionary of dictionary
        DESCRIPTION. Vertex network.
    cells : TYPE, int
        DESCRIPTION. Number of cells.
    vertices : TYPE, int
        DESCRIPTION. Number of vertices.
    k : TYPE, int
        DESCRIPTION. Solution index.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    tot_area : TYPE, float
        DESCRIPTION. Total area of the neural plate.
    tot_peri : TYPE, float
        DESCRIPTION. Total perimeter of the neural plate.
    av_area : TYPE, float
        DESCRIPTION. Average area of the cells in the neural plate.
    av_peri : TYPE, float
        DESCRIPTION. Average perimeter of the cells in the neural plate.
    U_tot_np : TYPE, float
        DESCRIPTION. Total potential energy of the neural plate.
    U_total : TYPE, float
        DESCRIPTION. Total potential energy of the neural plate and surface ectoderm.

    Returns
    -------
    None.

    '''
    
    with open(f'{result_dir}\\{directory}\\VertexPos_all_samples.txt', 'a') as f:
        f.write('Timestep %.3f\n'%(k*dt))
        for j in range(vertices):
            f.write(str(VertexGraph.nodes[j]['pos'])+' ')
        f.write('\n\n')
    
    with open(f'{result_dir}\\{directory}\\Area_all_samples.txt', 'a') as f:
        f.write('Timestep %.3f\n'%(k*dt))
        for i in range(cells):
            f.write(str(CellGraph.nodes[i]['area'])+' ')
        f.write('\n\n')
    
    with open(f'{result_dir}\\{directory}\\Peri_all_samples.txt', 'a') as f:
        f.write('Timestep %.3f\n'%(k*dt))
        for i in range(cells):
            f.write(str(CellGraph.nodes[i]['peri'])+' ')
        f.write('\n\n')
        
    with open(f'{result_dir}\\{directory}\\AreaNat_all_samples.txt', 'a') as f:
        f.write('Timestep %.3f\n'%(k*dt))
        for i in range(cells):
            f.write(str(CellGraph.nodes[i]['n_area'])+' ')
        f.write('\n\n')
        
    with open(f'{result_dir}\\{directory}\\PeriNat_all_samples.txt', 'a') as f:
        f.write('Timestep %.3f\n'%(k*dt))
        for i in range(cells):
            f.write(str(CellGraph.nodes[i]['n_peri'])+' ')
        f.write('\n\n')
        
    with open(f'{result_dir}\\{directory}\\CaConc_all_samples.txt', 'a') as f:
        f.write('Timestep %.3f\n'%(k*dt))
        for i in range(cells):
            f.write(str(CellGraph.nodes[i]['Ca_conc'])+' ')
        f.write('\n\n')
        
    with open(f'{result_dir}\\{directory}\\CaAmp_all_samples.txt', 'a') as f:
        f.write('Timestep %.3f\n'%(k*dt))
        for i in range(cells):
            f.write(str(CellGraph.nodes[i]['Ca_amp'])+' ')
        f.write('\n\n')    
    
    with open(f'{result_dir}\\{directory}\\TotArea_all_samples.txt', 'a') as f:
        f.write(str(tot_area)+' ')
        
    with open(f'{result_dir}\\{directory}\\TotPeri_all_samples.txt', 'a') as f:
        f.write(str(tot_peri)+' ')
    
    with open(f'{result_dir}\\{directory}\\AreaAv_all_samples.txt', 'a') as f:
        f.write(str(av_area)+' ')
        
    with open(f'{result_dir}\\{directory}\\PeriAv_all_samples.txt', 'a') as f:
        f.write(str(av_peri)+' ')
        
    with open(f'{result_dir}\\{directory}\\TotEngNP_all_samples.txt', 'a') as f:
        f.write(str(U_tot_np)+' ')
        
    with open(f'{result_dir}\\{directory}\\TotEng_all_samples.txt', 'a') as f:
        f.write(str(U_total)+' ')


def read_CellVert(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lst : TYPE, list of list
        DESCRIPTION. List of vertices of each cell.

    '''
    
    with open(f'{read_dir}\\Cell_vertices.txt') as f:
        line = f.read()
    
    line = line.split('Cell ')
    line.pop(0)
    
    lst = []
    
    for i in range(len(line)):
        
        lst.append(line[i].replace(f'{i}','',1).strip().split(' '))
        lst[i] = [int(l) for l in lst[i]]
        
    return lst


def read_VtxCells(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lst : TYPE, list of list
        DESCRIPTION. List of cells sharing each vertex.

    '''
    
    with open(f'{read_dir}\\Vertex_cells.txt') as f:
        line = f.read()
    
    line = line.split('Vertex ')
    line.pop(0)
    
    lst = []
    
    for i in range(len(line)):
        
        lst.append(line[i].replace(f'{i}','',1).strip().split(' '))
        lst[i] = [int(l) for l in lst[i]]
        
    return lst


def read_AreaAv(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lines : TYPE, list
        DESCRIPTION. Timeseries of the average area of the neural plate cells.

    '''
    
    with open(f'{read_dir}\\AreaAv_all_samples.txt') as f:
        line = f.read()
    
    line = line.split()
    lines = [float(l) for l in line]

    return lines


def read_PeriAv(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lines : TYPE, list
        DESCRIPTION. Timeseries of the average perimeter of the neural plate cells.

    '''
    
    with open(f'{read_dir}\\PeriAv_all_samples.txt') as f:
        line = f.read()
    
    line = line.split()
    lines = [float(l) for l in line]

    return lines


def read_TotEng(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lines : TYPE, list
        DESCRIPTION. Timeseries of the total potential energy of the neural plate and surface ectoderm.

    '''
    
    with open(f'{read_dir}\\TotEng_all_samples.txt') as f:
        line = f.read()
    
    line = line.split()
    lines = [float(l) for l in line]

    return lines


def read_TotEngNP(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lines : TYPE, list
        DESCRIPTION. Timeseries of the total potential energy of the neural plate.

    '''
    
    with open(f'{read_dir}\\TotEngNP_all_samples.txt') as f:
        line = f.read()
    
    line = line.split()
    lines = [float(l) for l in line]

    return lines


def read_TotArea(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lines : TYPE, list
        DESCRIPTION. Timeseries of the total area of the neural plate.

    '''
    
    with open(f'{read_dir}\\TotArea_all_samples.txt') as f:
        line = f.read()
    
    line = line.split()
    lines = [float(l) for l in line]

    return lines


def read_TotPeri(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lines : TYPE, list
        DESCRIPTION. Timeseries of the total perimeter of the neural plate.

    '''
    
    with open(f'{read_dir}\\TotPeri_all_samples.txt') as f:
        line = f.read()
    
    line = line.split()
    lines = [float(l) for l in line]

    return lines


def read_SEcells(read_dir):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.

    Returns
    -------
    lines : TYPE, list
        DESCRIPTION. Indices of the SE cells.

    '''
    
    with open(f'{read_dir}\\SE_cells.txt') as f:
        line = f.read()
    
    line = line.split()
    lines = [int(l) for l in line]

    return lines


def read_CaConc(read_dir,dt,txtskip):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    txtskip : TYPE, int
        DESCRIPTION. Number of timesteps skipped while writing the data.

    Returns
    -------
    transposed_lst : TYPE, list of list
        DESCRIPTION. Timeseries of the Ca2+ concentration of all cells.

    '''
    
    with open(f'{read_dir}\\CaConc_all_samples.txt') as f:
        line = f.read()
        
    line = line.split('Timestep ')
    line.pop(0)
    
    lst = []
    
    for i in range(len(line)):
        
        lst.append(line[i].replace(f'{i*txtskip*dt:.3f}','',1).strip().split(' '))
        lst[i] = [int(l) for l in lst[i]]
        
    transposed_lst = [[row[i] for row in lst] for i in range(len(lst[0]))]      
    
    return transposed_lst


def read_CaAmp(read_dir,dt,txtskip):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    txtskip : TYPE, int
        DESCRIPTION. Number of timesteps skipped while writing the data.

    Returns
    -------
    transposed_lst : TYPE, list of list
        DESCRIPTION. Timeseries of the Ca2+ amplitude of all cells.

    '''
    
    with open(f'{read_dir}\\CaAmp_all_samples.txt') as f:
        line = f.read()
        
    line = line.split('Timestep ')
    line.pop(0)
    
    lst = []
    
    for i in range(len(line)):
        
        lst.append(line[i].replace(f'{i*txtskip*dt:.3f}','',1).strip().split(' '))
        lst[i] = [float(l) for l in lst[i]]
        
    transposed_lst = [[row[i] for row in lst] for i in range(len(lst[0]))]      
    
    return transposed_lst


def read_VtxPos(read_dir,dt,txtskip):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    txtskip : TYPE, int
        DESCRIPTION. Number of timesteps skipped while writing the data.

    Returns
    -------
    transposed_lst : TYPE, list of list
        DESCRIPTION. Timeseries of the positions of all vertices.

    '''
    
    with open(f'{read_dir}\\VertexPos_all_samples.txt') as f:
        line = f.read()
        
    line = line.split('Timestep ')
    line.pop(0)
    
    lst = []
    
    for i in range(len(line)):

        lst.append(line[i].replace(f'{i*txtskip*dt:.3f}','',1).strip().split(') ('))
        lst[i][0] = lst[i][0].replace('(','').strip()
        lst[i][-1] = lst[i][-1].replace(')','').strip()
    
        lst[i] = [l.split(', ') for l in lst[i]]
    
        for j in range(len(lst[i])):
            lst[i][j] = [float(t) for t in lst[i][j]]
    
    transposed_lst = [[row[i] for row in lst] for i in range(len(lst[0]))]      
    
    return transposed_lst


def read_Pos(vtx_ind,soln_ind,read_dir,dt,txtskip):
    '''

    Parameters
    ----------
    vtx_ind : TYPE, int
        DESCRIPTION. Index of the vertex.
    soln_ind : TYPE, int
        DESCRIPTION. Index of the solution.
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    txtskip : TYPE, int
        DESCRIPTION. Number of timesteps skipped while writing the data.

    Returns
    -------
    xPos : TYPE, float
        DESCRIPTION. x-coordinate of the vertex at that timestep.
    yPos : TYPE, float
        DESCRIPTION. y-coordinate of the vertex at that timestep.

    '''
    
    j = vtx_ind
    k = soln_ind
    
    xPos = 0
    yPos = 0
    
    with open(f'{read_dir}\\VertexPos_all_samples.txt') as f:
        line = True
        
        while line:
            
            line = f.readline()
            if line == f'Timestep {k*txtskip*dt:.3f}\n':
                line = f.readline()
                line = line.strip().split(') (')
                line[0] = line[0].replace('(','').strip()
                line[-1] = line[-1].replace(')','').strip()
                line = [l.split(', ') for l in line]
                xPos = float(line[j][0])
                yPos = float(line[j][1])
                break
                
            else:
                continue
    
    return (xPos,yPos)


def generate_figs(read_dir,cells,hs,x_vc,y_vc,dt,An_min,Pn_min,time_slns,txtskip):
    '''

    Parameters
    ----------
    read_dir : TYPE, string
        DESCRIPTION. File path of the text file to be read.
    cells : TYPE, int
        DESCRIPTION. Number of cells.
    hs : TYPE, float
        DESCRIPTION. Edge length of regular hexagonal cell.
    x_vc : TYPE, float
        DESCRIPTION. x-coordinate of the initial position of the centre of the hexagonal lattice.
    y_vc : TYPE, float
        DESCRIPTION. y-coordinate of the initial position of the centre of the hexagonal lattice.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    An_min : TYPE, float
        DESCRIPTION. Minimum value limit for the cell natural area.
    Pn_min : TYPE, float
        DESCRIPTION. Minimum value limit for the cell natural perimeter.
    time_slns : TYPE, int
        DESCRIPTION. Total number of timesteps.
    txtskip : TYPE, int
        DESCRIPTION. Number of timesteps skipped while writing the data.

    Returns
    -------
    None.

    '''
    
    #VtxCells = read_VtxCells(read_dir)
    SEcells  = read_SEcells(read_dir)
    TotArea  = read_TotArea(read_dir)
    TotPeri  = read_TotPeri(read_dir)
    AreaAv   = read_AreaAv(read_dir)
    PeriAv   = read_PeriAv(read_dir)
    TotEngNP = read_TotEngNP(read_dir)
    TotEng   = read_TotEng(read_dir)
    
    if len(TotEng) <= 54000: 
        # 54000 timesteps is the highest value I've tested for which VertexPos_all_samples.txt can be read by read()
        
        CellVert = read_CellVert(read_dir)
        CaAmp    = read_CaAmp(read_dir,dt,txtskip)
        VtxPos   = read_VtxPos(read_dir,dt,txtskip)
        
        #Fig0 = plt.figure(figsize=(20,20))
        
        # For the colorbar
        # ----------------
        cmap = plt.get_cmap('YlGn', 100)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Read vertex position for large files
        # ------------------------------------
        # VtxPos = read_Pos(j,k,read_dir,dt,txtskip)
        
        dq = (time_slns/txtskip)/5
        
        f_snap = [0,int(1*dq),int(2*dq),int(3*dq),int(4*dq),int(5*dq)-1]
        
        for k in f_snap:
            
            Fig0, ax0 = plt.subplots(figsize=(20, 16))
            
            for i in range(cells):
                X = []
                Y = []
            
                for j in CellVert[i]:
                    X.append(VtxPos[j][k][0])
                    Y.append(VtxPos[j][k][1])     
                
                Ca_flash = CaAmp[i][k]
                SE_cell = 1 if i in SEcells else 0
                HexPlot(X,Y,ax0,cmap,int(100*Ca_flash),SE_cell)
            
            plt.xlim([0,2*x_vc])
            plt.ylim([0,2*y_vc])
            
            plt.text(0+0.1*hs,0+0.2*hs,str(round(k*txtskip*dt,2)),fontsize=40) 
            #plt.text(0+0.1*hs,0+0.2*hs,str(round(k*dt,2)),fontsize=22,bbox = dict(facecolor = 'green', alpha = 0.5, edgecolor = 'black'))
            
            cb0 = plt.colorbar(mappable=sm,ax=ax0)
            cb0.ax.tick_params(labelsize=20)
            cb0.set_label('$Ca^{2+}$ flash amplitude',fontsize=20)
            
            #plt.xticks([0,2,4,6,8,10,12,14,16,18],[0,2,4,6,8,10,12,14,16,18],fontsize=30)
            
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.savefig(f'{read_dir}\\Tissue_t'+str(int(k*txtskip*dt))+'.png',dpi=150)
            plt.clf()
        
        plt.close(Fig0)
    
    t_sol = dt*txtskip*np.linspace(0,len(TotEng)-1,len(TotEng))
    
    Fig1 = plt.figure()        
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(t_sol, AreaAv, '-k', label='Average NP cell area wrt time')
    plt.axhline(An_min, color='red', linestyle='--')
    plt.text(10,An_min-0.1*hs,'An_min = '+str(An_min),fontsize=12)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('Av. area', fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f'{read_dir}\\AvgArea.png', dpi=150)
    plt.close(Fig1)
    
    Fig2 = plt.figure()        
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(t_sol, PeriAv, '-k', label='Average NP cell perimeter wrt time')
    plt.axhline(Pn_min, color='red', linestyle='--')
    plt.text(10,Pn_min-0.3*hs,'Pn_min = '+str(Pn_min),fontsize=12)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('Av. perimeter', fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f'{read_dir}\\AvgPeri.png', dpi=150)
    plt.close(Fig2)
    
    Fig3 = plt.figure()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(t_sol, TotEng, '-k', label='Energy of the NP and SE wrt time')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('Tissue total energy', fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f'{read_dir}\\TotEner.png', dpi=150)
    plt.close(Fig3)
    
    Fig4 = plt.figure()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(t_sol, TotArea, '-k', label='NP area wrt time')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('NP area', fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f'{read_dir}\\TotArea.png', dpi=150)
    plt.close(Fig4)
    
    Fig5 = plt.figure()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(t_sol, TotPeri, '-k', label='NP perimeter wrt time')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('NP perimeter', fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f'{read_dir}\\TotPeri.png', dpi=150)
    plt.close(Fig5)
    
    Fig6 = plt.figure()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(t_sol, TotEngNP, '-k', label='Energy of the NP wrt time')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('NP energy', fontsize=12)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f'{read_dir}\\TotEnerNP.png', dpi=150)
    plt.close(Fig6)
    

def cell_area(X,Y):
    '''
    
    Parameters
    ----------
    X : TYPE, list of float
        DESCRIPTION. The x-coordinates of cell vertices in anti-clockwise order.
    Y : TYPE, list of float
        DESCRIPTION. The y-coordinates of cell vertices in anti-clockwise order.

    Returns
    -------
    TYPE, float
        DESCRIPTION. Area of cell. 

    '''
    
    coords = np.zeros((6,2))
    coords[:,0] = X
    coords[:,1] = Y
    
    return Polygon(coords).area

    
def cell_peri(X,Y):
    '''

    Parameters
    ----------
    X : TYPE, list of float
        DESCRIPTION. The x-coordinates of cell vertices in anti-clockwise order.
    Y : TYPE, list of float
        DESCRIPTION. The y-coordinates of cell vertices in anti-clockwise order.

    Returns
    -------
    perimeter : TYPE, float
        DESCRIPTION. Perimeter of cell.

    '''
    
    perimeter = 0
    
    for i in range(-1,5):
        
        dist = np.sqrt(np.power((X[i]-X[i+1]),2)+np.power((Y[i]-Y[i+1]),2))
        perimeter = perimeter + dist
        
    return perimeter


def enfunc(Ka,A,An,Kp,P,Pn,T,AdC,Ln):
    '''

    Parameters
    ----------
    Ka : TYPE, list of float
        DESCRIPTION. Area elasticity coefficients of the cells associated with 'this' vertex.
    A : TYPE, list of float
        DESCRIPTION. Areas of cells associated with 'this' vertex. 
    An : TYPE, list of float
        DESCRIPTION. Natural areas of cells associated with 'this' vertex.
    Kp : TYPE, list of float
        DESCRIPTION. Perimeter elasticity coefficients of the cells associated with 'this' vertex.    
    P : TYPE, list of float
        DESCRIPTION. Perimeters of cells associated with 'this' vertex.
    Pn : TYPE, list of float
        DESCRIPTION. Natural perimeters of cells associated with 'this' vertex.
    T : TYPE, list of float
        DESCRIPTION. Line tensions of the cell junctions connecting to 'this' vertex.
    AdC : TYPE, list of float
        DESCRIPTION. Coefficients for the adhesion functions of the cell junctions connecting to 'this' vertex.    
    Ln : TYPE, list of float
        DESCRIPTION. Lengths of the cell junctions connecting to 'this' vertex.

    Returns
    -------
    TYPE, float
        DESCRIPTION. Energy functional (only the terms relevant to 'this' vertex).

    '''
    
    area_func = 0
    peri_func = 0
    tens_func = 0
    
    for n in range(len(A)):
        
        area_func = area_func + 0.5*Ka[n]*An[n]*np.power((A[n]/An[n]-1),2) 
        peri_func = peri_func + 0.5*Kp[n]*Pn[n]*np.power((P[n]/Pn[n]-1),2)
        
    for n in range(len(T)):
        
        tens_func = tens_func + T[n]*Ln[n] - (AdC[n]/4.69)*np.arctan(31.32*Ln[n])
        # Note: the adhesion function coefficients only need to change if we change (Pn_min/6)
        
    return area_func+peri_func+tens_func


def Hill_derivative(n,A,B,tau,t):
    '''

    Parameters
    ----------
    n : TYPE, int
        DESCRIPTION. Hill coefficient of the nth order Hill function describing the time evolution of xi.
    A : TYPE, float
        DESCRIPTION. The absolute maximum value that xi can have over the course of the simulation.
    B : TYPE, float
        DESCRIPTION. A constant of the nth order Hill function describing the time evolution of xi. 
    tau : TYPE, float
        DESCRIPTION. Time duration of a Ca2+ flash.
    t : TYPE, float
        DESCRIPTION. Time elapsed since the start of the Ca2+ spike.

    Returns
    -------
    der : TYPE, float
        DESCRIPTION. Derivative of the nth order Hill function describing the time evolution of xi.

    '''

    der = (1/tau)*(n*A*B*(t/tau)**(n-1))/((B+(t/tau)**n)**2)        

    return der


def Hill_fcn(n,A,B,t):
    '''

    Parameters
    ----------
    n : TYPE, int
        DESCRIPTION. Hill coefficient of the nth order Hill function.
    A : TYPE, float
        DESCRIPTION. The maximum value that the Hill function can have.
    B : TYPE, float
        DESCRIPTION. Constant that determines the position of the Hill function curve along the t-axis. 
    t : TYPE, float
        DESCRIPTION. The independent variable.

    Returns
    -------
    fcnval : TYPE, float
        DESCRIPTION. Value of the Hill function evaluated for t.

    '''

    fcnval = (A*(t**n))/(B+(t**n))      

    return fcnval


def Hill_constant(n,A,f0,t0):
    '''

    Parameters
    ----------
    n : TYPE, int
        DESCRIPTION. Hill coefficient of the nth order Hill function.
    A : TYPE, float
        DESCRIPTION. The maximum value that the Hill function can have.
    f0 : TYPE, float
        DESCRIPTION. Value of the nth order Hill function for t0.
    t0 : TYPE, float
        DESCRIPTION. A constant value.

    Returns
    -------
    B : TYPE, float
        DESCRIPTION. Constant that determines the position of the Hill function curve along the t-axis for the specified values of f0 and t0.

    '''
    
    B = ((A/f0)-1)*t0**n
    
    return B
        

def fwd_euler(mu,dt,qti,dU,dr):
    '''

    Parameters
    ----------
    mu : TYPE, float
        DESCRIPTION. Viscosity coefficient.
    dt : TYPE, float
        DESCRIPTION. Size of timestep.
    qti : TYPE, float
        DESCRIPTION. Vertex position coordinate at current timestep. 
    dU : TYPE, float
        DESCRIPTION. Change in tissue energy when vertex displaced by dr (either x or y coordinate).
    dr : TYPE, float
        DESCRIPTION. Vertex displacement - equal to spatial stepsize.

    Returns
    -------
    nqti : TYPE, float
        DESCRIPTION. Vertex position coordinate at next timestep.

    '''
    
    nqti = qti + (-1/mu)*dt*(dU/dr)
    
    return nqti


def HexPlot(X,Y,ax,cmap,Ca_level=0,SE=0):
    '''

    Parameters
    ----------
    X : TYPE, list of float
        DESCRIPTION. The x-coordinates of cell vertices in anti-clockwise order.
    Y : TYPE, list of float
        DESCRIPTION. The y-coordinates of cell vertices in anti-clockwise order.
    ax : TYPE, object of Axes class
        DESCRIPTION. Axis object to which we assign the subplot depicting the tissue at some timestep.
    cmap : TYPE, int (probably)
        DESCRIPTION. Colormap for the figure. Could set default to cm.get_cmap('YlGn',100).
    Ca_level : TYPE, float
        DESCRIPTION. Ca2+ level of the cell. The default is 0.
    SE : TYPE, int
        DESCRIPTION. Is the cell a surface ectoderm cell. The default is 0 i.e. false.

    Returns
    -------
    None.

    '''
    
    # X and Y to be lists of equal length, length = 6 for hexagonal cells
    # value must be in cyclic order, can be anticlockwise or clockwise
    
    for i in range(0, len(X)):
        
        ax.plot([X[i-1],X[i]],[Y[i-1],Y[i]], 'k-')
    
    fc = 'brown' if SE == 1 else cmap(Ca_level)
    ax.fill(X,Y,facecolor=fc)


def tissue_area(X,Y,vnum):
    '''

    Parameters
    ----------
    X : TYPE, list of float
        DESCRIPTION. The x-coordinates of boundary vertices in anti-clockwise order.
    Y : TYPE, list of float
        DESCRIPTION. The y-coordinates of boundary vertices in anti-clockwise order.
    vnum : TYPE, int
        DESCRIPTION. Number of boundary vertices.

    Returns
    -------
    TYPE, float
        DESCRIPTION. Area of tissue. 

    '''
    
    coords = np.zeros((vnum,2))
    coords[:,0] = X
    coords[:,1] = Y
    
    return Polygon(coords).area


def clearQueue(q,qf):
    '''

    Parameters
    ----------
    q : TYPE, object of Queue class of multiprocessing package
        DESCRIPTION. Buffer queue holding the progress percentage of each parallel process.
    qf : TYPE, object of Value class of multiprocessing package
        DESCRIPTION. Flag that reflects the full(1)/not full(0) status of q.

    Returns
    -------
    None.

    '''
    
    qf.value = 1
    try:
        while True:
            q.get(False)            
    except Empty:
        qf.value = 0


def main():
    '''
    
    Definitions of the User Input variables:-
        layers:   Layers of the hexagon-shaped lattice. layers = 0 --> 1 cell.
        hs:       Edge length of regular hexagonal cell.
        k_se:     Multiplicative factor for SE cell elasticities.
        simtime:  Total time over which solutions are being computed.
        samples:  Number of simulation runs for each parameter value.
        num_proc: Number of parallel processes.

    Returns
    -------
    None.

    '''
    
    tic = time.time() 
    
    # ------------------------------------------------------------------------
    #                              User Input
    # ------------------------------------------------------------------------
    
    NP_layers = 9
    SE_layers = 1
    hs   = 0.55
    k_se = 0.55
    simtime = 6000
    
    samples = 1
    
    #num_proc = mp.cpu_count()
    num_proc = 10
    
    # ------------------------------------------------------------------------
    
    dx = 1e-6
    dy = 1e-6
    
    # ------------------------------------------------------------------------
    
    result_dir = create_results_dir()
    
    CellGraph, VertexGraph, Boundary, NP_Boundary, TissueCentre_x, TissueCentre_y = generate_tissue(NP_layers,SE_layers,hs) 
    
    cells = len(CellGraph.nodes())
    vertices = len(VertexGraph.nodes())
    # reg_hex_area = 1.5*math.sqrt(3)*(hs**2) # area of regular hexagon with edge length 'hs'
    
    NP_cells = 1 + 6*int((NP_layers/2)*(NP_layers+1))
    SE_cells = cells - NP_cells
    
    # ------------------------------------------------------------------------
    #                          Terminal display
    # ------------------------------------------------------------------------
    
    print('-------------------')
    print('| Parameter index |')
    print('-------------------')
    print('---------------------------------------------------------')
    print('0 : Fraction of cells activated initially (cf_ini)')
    print('1 : Flash per min per cell (fpm)')
    print('2 : Duty ratio (duty)')
    print('3 : Timestep (dt)')
    print('4 : Natural Area shrinkage - Rate constant (Ca)')
    print('5 : Natural Area shrinkage - Threshold constant (Wa)')
    print('6 : Natural Area - Initial value (An_ini)')
    print('7 : Natural Perimeter - Initial value (Pn_ini)')
    print('8 : Natural Area - Minimum value (An_min)')
    print('9 : Natural Perimeter - Minimum value (Pn_min)')
    print('10 : Ca2+ flash amplitude (flash_amp)')
    print('11 : Ca2+ flash duration (flash_time)')
    print('---------------------------------------------------------')
    print('')
    
    # ------------------------------------------------------------------------
    
    default_parameters, sweep_list = set_sim_params(result_dir,cells,NP_cells,SE_cells,hs,k_se,simtime)
    num_par = len([pv for pn in sweep_list for pv in pn])
    
    if num_par == 0:
        
        print('Running simulation for default parameter values')
        
        directory = 'Default Parameters'
        path = os.path.join(result_dir,directory)
        try:
            os.mkdir(path)
            print("Directory %s created" %directory)
        except OSError as error:
            print(error)

        err_flag = simulate_VM(result_dir,directory,CellGraph,VertexGraph,Boundary,NP_Boundary,TissueCentre_x,TissueCentre_y,cells,vertices,simtime,hs,k_se,dx,dy,default_parameters)
        toc = time.time()
        print(f'Simulation complete. Time taken: {toc-tic}s')
        print('---------------------------------------------------')
        if err_flag == 0:
            print('Simulation successful for default parameter values.')
        else:
            print('Simulation failed for default parameter values.')
        print('---------------------------------------------------')
        
    else:
        
        if samples == 1:
            print(f'Running simulations for {num_par} parameters in total')
        
        else:
            print(f'Running {samples} simulations each for {num_par} parameters. Total no. of simulations: {samples*num_par}')
        
        par_ind_val_list = []
        err_param = []
        
        for par_ind in range(len(sweep_list)):
            
            for par in sweep_list[par_ind]:
                
                for s_no in range(samples):
                    
                    par_ind_val_list.append((par_ind,par,s_no))
                
        q_tstep = mp.Queue()
        q_errp  = mp.Queue()
        q_full  = mp.Value('i',0)
        
        out_buff = [[i,0] for i in range(num_proc)] if len(par_ind_val_list) >= num_proc else [[i,0] for i in range(len(par_ind_val_list))]
         
        samp_no = ''
        par_ctr = 0
         
        while True:
            
            if len(mp.active_children()) < num_proc and par_ctr < len(par_ind_val_list):
                
                par_ind = par_ind_val_list[par_ctr][0]
                par = par_ind_val_list[par_ctr][1]
                s_no = par_ind_val_list[par_ctr][2]
                
                if samples > 1:
                    samp_no = ' Sample'+str(s_no)
                
                directory = f'Parameter {par_ind} - ' + str(par).replace('.','_') + samp_no
                path = os.path.join(result_dir,directory)
                try:
                    os.mkdir(path)
                except OSError as error:
                    print(error)
                
                mp.Process(target = simulate_VM, args = (result_dir,directory,CellGraph,VertexGraph,Boundary,NP_Boundary,TissueCentre_x,TissueCentre_y,cells,vertices,simtime,hs,k_se,dx,dy,default_parameters,False,par_ctr,par_ind,par,s_no,q_tstep,q_full,q_errp)).start()
                par_ctr += 1
                    
            try:
                q_get = q_tstep.get(timeout=5) # Recommended: larger number of cells --> higher timeout
                
            except Empty:
                if len(mp.active_children()) == 0 and par_ctr == len(par_ind_val_list):
                    if not q_errp.empty():
                        while not q_errp.empty():
                            err_param.append(q_errp.get(False))
                    break
                else:
                    continue
            
            proc_ind = q_get[0]
            proc_val = q_get[1]
            out_buff_ind = [out_buff[i][0] for i in range(len(out_buff))]
            p_ind = out_buff_ind.index(proc_ind) if proc_ind in out_buff_ind else None
            if p_ind is not None:
                out_buff[p_ind][1] = proc_val
            else:
                out_buff_val = [out_buff[i][1] for i in range(len(out_buff))]
                p_ind = out_buff_val.index(100) if 100 in out_buff_val else None
                if p_ind is not None:
                    out_buff[p_ind][0] = proc_ind
                    out_buff[p_ind][1] = proc_val
                    print("                                                               ",end="")
                    print("                                                               ",end="")
                    print("                                                               ",end="")
                    print("",end='\r')
            
            #'''
            # - Display progress of all processes -
            
            out_line = ""
            
            for ob in out_buff:
                
                samp_br = ""
                if samples > 1:
                    samp_br = f"({par_ind_val_list[ob[0]][2]})"
                out_line = out_line + f"P {par_ind_val_list[ob[0]][0]} - {par_ind_val_list[ob[0]][1]}" + samp_br + f": {ob[1]}%, "
            
            print(out_line,end='\r')
            
            # -------------------------------------
            #'''
            
            '''
            # - Only display progress of oldest and newest processes in the current batch -
            
            out_buff_ind = [out_buff[i][0] for i in range(len(out_buff))]
            p_min = out_buff_ind.index(min(out_buff_ind))
            p_max = out_buff_ind.index(max(out_buff_ind))
            samp_br_min = ""
            samp_br_max = ""
            if samples > 1:
                samp_br_min = f"({par_ind_val_list[out_buff[p_min][0]][2]})"
                samp_br_max = f"({par_ind_val_list[out_buff[p_max][0]][2]})"
            out_line_1 = f"P {par_ind_val_list[out_buff[p_min][0]][0]} - {par_ind_val_list[out_buff[p_min][0]][1]}" + samp_br_min + f": {out_buff[p_min][1]}%, "
            out_line_2 = f"P {par_ind_val_list[out_buff[p_max][0]][0]} - {par_ind_val_list[out_buff[p_max][0]][1]}" + samp_br_max + f": {out_buff[p_max][1]}%"
            print(out_line_1 + out_line_2,end='\r')    
            
            # -----------------------------------------------------------------------------
            '''
            
            if not q_errp.empty():
                try:
                    err_param.append(q_errp.get(False))
                except Empty:
                    continue
        
        q_tstep.close()
        q_errp.close()
        
        with open(f'{result_dir}\\ErrorParams.txt', 'w') as f:
            for erp in err_param:
                f.write(str(erp)+'\n')
        
        print("                                                               ",end="")
        print("                                                               ",end="")
        print("                                                               ",end="")
        print("",end='\r')
        
        toc = time.time()
        print(f'\nParameter sweep completed. Time taken: {toc-tic}s')
        print('--------------------------------------------')
        print('Sweep failed for following parameter values:')
        print('(index, value)')
        for erp in err_param:
            print(erp)
        print('--------------------------------------------')

    
# ----------------------------------------------------------------------------    


if __name__ == '__main__':
    main()
    

"""

#-----------------------------------------------------------------------------------------------

# Use if calling different scripts
#  import importlib
#  importlib.reload(scriptname)

#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
#                             Position randomizer - Vertex Graph
#-----------------------------------------------------------------------------------------------


for i in range(6*pow(layers+1,2)):
    
    vtx_xPos = VertexGraph_def.nodes[i]['pos'][0][0]
    vtx_yPos = VertexGraph_def.nodes[i]['pos'][0][1]
    
    rad = hs/2*np.random.random_sample()
    ang = math.pi*np.random.random_sample()
    
    vtx_xPos = vtx_xPos + rad*np.cos(ang)
    vtx_yPos = vtx_yPos + rad*np.sin(ang)
    
    VertexGraph_def.nodes[i]['pos'][0] = (vtx_xPos,vtx_yPos)


#-----------------------------------------------------------------------------------------------
#                               NOTE: List of boundary vertices 
#-----------------------------------------------------------------------------------------------


# Note: This code uses VertexGraph_def, the model uses deepcopies (instances... kind of... ) during runtime
#  I can get away with this now because I don't implement topological transitions
#  But if graph topology changes during runtime, Boundary should be included 
#  within the Model Physics block of code and update accordingly

# Addendum #1: Not for now, but if we're dealing with a more complex tissue, where forces are different
#  for every cell-cell junction, we should make use of the graph edges i.e. the line tension can be
#  set as an attribute of the corresponding edges and be read from there during computations 

# Addendum #2: The following algorithm takes advantage of the fact that the cells are hexagonal.
#  For more complex tissues where cells can be any n-sided polygon, finding boundary vertices is
#  more challenging. Without using certain restricting conditions or "hacks", it's probably impossible
#  to find boundary vertices because, if you think in terms of just topology, 
#  you can't define a boundary for the vertex graph!!
#  For instance, a single cell (6 nodes) could be said to 'bound' all of the other nodes, in terms of topology.

#  The best approach is then to define the boundary vertices or boundary edges at the time of
#  tissue initialization when the shape of cells (and tissue) are known and their properties can be
#  taken advantage of. During model runtime, the list of boundary vertices or edges can be updated
#  as required.


#-----------------------------------------------------------------------------------------------
#                                  NOTE: Function definitions
#-----------------------------------------------------------------------------------------------


# default colormap for this function, colormap divided into 100 gradations
col_map = cm.get_cmap('YlGn',100) 
def HexPlot(X,Y,cmap=col_map,Ca_level=0): 
    # X and Y to be lists of equal length, length = 6 for hexagonal cells
    # value must be in cyclic order, can be anticlockwise or clockwise
    #plt.plot(X, Y, 'o');
    for i in range(0, len(X)):
        plt.plot([X[i-1],X[i]],[Y[i-1],Y[i]], 'k-')
    plt.fill(X,Y,facecolor = cmap(Ca_level))


#-----------------------------------------------------------------------------------------------
#                                     NOTE: Function call
#-----------------------------------------------------------------------------------------------    


# clearQueue(q,qf)
# Note: an object (instance of a class) is passed by reference


#-----------------------------------------------------------------------------------------------
#                                  NOTE: Parallel processing
#-----------------------------------------------------------------------------------------------


# num_proc = 8 

# this value should be the same as the number of logical cores i.e. 8 
(My system: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx, 2100 Mhz, 4 Core(s), 8 Logical Processor(s))
# based on what I saw with my system, this reduces program runtime by 4 (the number of physical cores)    


#-----------------------------------------------------------------------------------------------
"""    
    
    