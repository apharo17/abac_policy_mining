import os
import sys
sys.path.append(os.path.abspath('..'))
from efficient_apriori import apriori
#Further info: https://programmerclick.com/article/97871687554/
#TODO: Test in other operative systems

import numpy as np
import igraph as ig
import src.acgraph as acg
from src import utils as ut
from src.biclique import Biclique
import copy



def create_bcgraph(subbcs, B, th):
    #gur is the original access control graph
    #subbcs is a list of Biclique objects
    #th is the threshold to filter the edges of the bcgraph
    #B is the weight matrix

    #Create bcgraph from B
    bcgraph = ig.Graph.Adjacency((B >= th).tolist())

    #Add the bcid to nodes
    for bcidx in range(len(subbcs)):
        bcgraph.vs[bcidx]['bcid'] = subbcs[bcidx].get_id()

    #Add edge weights
    for e in bcgraph.es:
        bcvidx = e.source
        bcvidx2 = e.target
        e['weight'] = B[bcvidx,bcvidx2]

    return bcgraph



def get_weight_matrix(gur, subbcs, verbose=False):

    #print('Computing...')

    #Auxiliar mappings
    usrid_to_vidx = dict() #1
    resid_to_vidx = dict() #2
    for v in gur.vs:
        vidx = v.index
        if v['type'] == False:
            usrid = int(v['name'][6:])
            usrid_to_vidx[usrid] = vidx
        else:
            resid = int(v['name'][6:])
            resid_to_vidx[resid] = vidx


    #Mapping of vertices from the ACG to the bcgraph
    vidx_to_bcidxs = dict() #1
    bcidx_to_vidxs = [] #2

    for bcidx,bc in enumerate(subbcs):
        vidxs = _get_vidxs(bc, usrid_to_vidx, resid_to_vidx)
        bcidx_to_vidxs.append(vidxs) #2

        for vidx in vidxs:
            if not vidx in vidx_to_bcidxs:
                vidx_to_bcidxs[vidx] = []
            vidx_to_bcidxs[vidx].append(bcidx) #1


    #Compute betavcs
    numv = len(gur.vs)
    numbcs = len(subbcs)
    Betavc = np.zeros((numv,numbcs)) #This is a mapping vidx_bcidx_to_betavc

    for vidx,bcidxs in vidx_to_bcidxs.items():
        for bcidx in bcidxs:
            bc_vidxs = bcidx_to_vidxs[bcidx]
            Betavc[vidx,bcidx] = _get_betavc(gur, vidx, bc_vidxs, vidx_to_bcidxs)



    #Compute alphavcs
    Alphavc = np.zeros((numv,numbcs)) #This is a mapping vidx_bcidx_to_alphavc
    for vidx,bcidxs in vidx_to_bcidxs.items():
        for bcidx in bcidxs:
            Alphavc[vidx,bcidx] = _get_alphavc(vidx, bcidx, Betavc, vidx_to_bcidxs)


    #Compute B
    numbcs = len(subbcs)
    B = np.zeros((numbcs,numbcs))



    for bcidx in range(len(subbcs)-1):
        if verbose:
            print(bcidx) #For Debug
        bc = subbcs[bcidx]
        bc_vidxs = bcidx_to_vidxs[bcidx]

        for bcidx2 in range(bcidx+1,len(subbcs)):
            bc2 = subbcs[bcidx2]
            bc_vidxs2 = bcidx_to_vidxs[bcidx2]

            B[bcidx,bcidx2] = _get_weight(gur, bcidx, bc_vidxs, bcidx2, bc_vidxs2, Alphavc)


    #Delete self-loops and redundant edges
    for bcidx in range(numbcs):
        B[bcidx,bcidx] = 0.0
    for bcidx in range(numbcs-1):
        for bcidx2 in range(bcidx+1,numbcs):
            B[bcidx2,bcidx] = 0.0

    return B


def _get_vidxs(bc, usrid_to_vidx, resid_to_vidx):
    usr_vidxs = {usrid_to_vidx[usrid] for usrid in bc.get_usrids()}
    res_vidxs = {resid_to_vidx[resid] for resid in bc.get_resids()}
    return list(usr_vidxs | res_vidxs)


def _get_betavc(gur, vidx, bc_vidxs, vidx_to_bcidxs):

    bcidxset = set(vidx_to_bcidxs[vidx])
    summ = 0
    for vidx2 in bc_vidxs:
        if gur.are_connected(vidx, vidx2):
            bcidxset2 = set(vidx_to_bcidxs[vidx2])
            summ += 1/(len(bcidxset&bcidxset2))
    return summ


def _get_alphavc(vidx, bcidx, Betavc, vidx_to_bcidxs):
    summ = 0
    for bcidx2 in vidx_to_bcidxs[vidx]:
        summ += Betavc[vidx,bcidx2]

    return Betavc[vidx,bcidx]/summ


def _get_weight(gur, bcidx, bc_vidxs, bcidx2, bc_vidxs2, Alphavc):
    summ = 0
    for bc_vidx in bc_vidxs:
        for bc_vidx2 in bc_vidxs2:
            if gur.are_connected(bc_vidx, bc_vidx2):
                alpha = Alphavc[bc_vidx,bcidx]
                alpha2 = Alphavc[bc_vidx2,bcidx2]
                summ += alpha*alpha2

    return summ



#-------------------------------------------------------------------------------

def check_auxtable(vidx, pattset, vidx_to_setpattern_list):
    #Check if vidx has associtated pattset in
    #the auxiliary table vidx_to_setpattern_list
    if vidx in vidx_to_setpattern_list:
        if pattset in vidx_to_setpattern_list[vidx]:
            return True
    return False


def update_auxtable(vidx, pattset, vidx_to_setpattern_list):
    #Append pattset in list of vidx in
    #the auxiliary table vidx_to_setpattern_list
    if vidx in vidx_to_setpattern_list:
        vidx_to_setpattern_list[vidx].append(pattset)
    else:
        vidx_to_setpattern_list[vidx] = [pattset]


#TODO: Falto lo siguiente
#Unir distintos pares s-v que tengan el mismo patron
#Los resultados en AZKAG se parecen mucho a los conseguidos con el primer approach
#de DFS

#Dado que compartir usuarios, ya implica que dos bicliques van a compartir
#patrones muy parecidos, el utilizar p_min<1 no aporta mucha compresion.

#Parece que se ve mas estetico el algoritmo si utilizamos BFS

#---Code to call dfs
#resid_to_bcpatterns = dict()
#for resid,bcgraph in resid_to_bcgraph_.items():
#    print('resid:', resid)
#    bcpatterns = bcg.dfs(bcgraph, bcid_to_bc, 1.0)
#    resid_to_bcpatterns[resid] = bcpatterns


#def dfs(bcgraph, bcid_to_bc, p_min=1.0):
def dfs(bcgraph, bcid_to_bc, sa=1):

    #Search bcpatterns in bcgraph that have empalme al menos de p_min

    bcpatterns = [] #list of patterns

    #---Init pattset auxiliary table---
    vidx_to_setpattern_list = dict()

    for s in bcgraph.vs:    #for each source s
        sidx = s.index
        sbcid = s['bcid']
        pattsource = bcid_to_bc[sbcid].get_pattern()

        count = 0 #It counts the number of patterns found with s

        for vidx in bcgraph.neighbors(s, mode='all'):
            v = bcgraph.vs[vidx]
            vbcid = v['bcid']
            pattv = bcid_to_bc[vbcid].get_pattern()

            #---Get reference pattern---
            pattset_ref = set(pattsource)&set(pattv)

            #if len(pattset_ref)/len(pattsource) >= p_min:
            if len(pattset_ref) >= sa:

                #Note: you can check for either vidx or sidx
                if check_auxtable(vidx, pattset_ref, vidx_to_setpattern_list) == False and check_auxtable(sidx, pattset_ref, vidx_to_setpattern_list) == False:

                    count += 1

                    #Add sidx to auxiliary table with pattset_ref
                    update_auxtable(sidx, pattset_ref, vidx_to_setpattern_list)
                    #Note: vidx with pattset_ref will be added in the visit function

                    #---Reset biclique list---
                    bcids_list=[]
                    bcids_list.append(sbcid)
                    #Note: sidx will be added in the visit function

                    #---Reset visit table---
                    vidx_to_isvisited = {v2.index:False for v2 in bcgraph.vs}
                    vidx_to_isvisited[sidx] = True
                    vidx_to_isvisited[vidx] = True

                    dfs_visit(vidx=vidx,
                        pattset_ref=pattset_ref,
                        bcids_list=bcids_list,
                        bcgraph=bcgraph,
                        bcid_to_bc=bcid_to_bc,
                        vidx_to_isvisited=vidx_to_isvisited,
                        vidx_to_setpattern_list=vidx_to_setpattern_list)


                    bcpatterns.append((list(pattset_ref), bcids_list))
                    #Note: here the pattern has list format



        if count == 0 and check_auxtable(sidx, set(pattsource), vidx_to_setpattern_list) == False:
            bcpatterns.append((pattsource, [sbcid]))

    return bcpatterns



def dfs_visit(vidx, pattset_ref, bcids_list, bcgraph, bcid_to_bc, vidx_to_isvisited, vidx_to_setpattern_list):
    #This traversal looks for vertices in bcgraph starting from vidx,
    #that contain the pattset_ref in a pattern subset.
    #Note pattset_ref is a pattern in set format
    #It returns: bcids_list

    v = bcgraph.vs[vidx]
    vbcid = v['bcid']
    pattv = bcid_to_bc[vbcid].get_pattern()

    if len(set(pattv)&pattset_ref) == len(pattset_ref):

        update_auxtable(vidx, pattset_ref, vidx_to_setpattern_list)

        bcids_list.append(vbcid)

        #Iterate over the neighbors
        for v2idx in bcgraph.neighbors(v, mode='all'):
            if not vidx_to_isvisited[v2idx]:
                vidx_to_isvisited[v2idx] = True
                dfs_visit(vidx=v2idx,
                pattset_ref=pattset_ref,
                bcids_list=bcids_list,
                bcgraph=bcgraph,
                bcid_to_bc=bcid_to_bc,
                vidx_to_isvisited=vidx_to_isvisited,
                vidx_to_setpattern_list=vidx_to_setpattern_list)
