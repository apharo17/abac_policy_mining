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


def _overlap_coeff(pattset1, pattset2):
    if len(pattset1) < len(pattset2):
        minlen = len(pattset1)
    else:
        minlen = len(pattset2)

    return len(pattset1&pattset2)/minlen


def _filter_closed(freq_itemsets):
    freq_itemsets_ = dict()

    levels = list(range(len(freq_itemsets)))

    level_to_keylen = list(freq_itemsets.keys())

    for level in levels[:-1]: #Don't check the leaves

        keylen = level_to_keylen[level]
        next_keylen = level_to_keylen[level+1]

        freq_itemsets_[keylen] = dict()

        for patttup1,freq1 in freq_itemsets[keylen].items():
            pattset1 = set(patttup1)
            flag = True

            #Check all super patterns
            for patttup2,freq2 in freq_itemsets[next_keylen].items():
                pattset2 = set(patttup2)
                oc = _overlap_coeff(pattset1, pattset2)
                #print(int(oc))
                if int(oc) == 1: #check pattset2 is a super pattern
                    if freq1 == freq2:
                        flag = False
                        break
            if flag:
                freq_itemsets_[keylen][patttup1] = freq1

    #For the leaves, just copy
    if len(levels) > 0:
        last_keylen = level_to_keylen[levels[-1]]
        freq_itemsets_[last_keylen] = freq_itemsets[last_keylen]

    return freq_itemsets_


def compute_avpatterns(gur, lmin=1, global_f=10, verbose=False):

    attnames = gur.vs[0].attribute_names()
    lengths = list(range(lmin,len(attnames)+1))

    #Get resids
    _,reslabels = acg.get_labels(gur, byweights=False)
    resids = acg.get_resids(reslabels)

    #Declare the return dictionaries
    resid_to_patterns = {resid:[] for resid in resids}
    resid_to_valsentries = {resid:None for resid in resids}


    for v in gur.vs:

        if v['type'] == True:
            vr = v

            resid = int(vr['name'][6:])
            if verbose:
                print(resid)

            requesters_usrids = []
            weights = []
            usrvalues_list = []

            for eidx in gur.incident(vr, mode='all'):
                e = gur.es[eidx]
                if e.target != vr.index:
                    vu = gur.vs[e.target]
                else:
                    vu = gur.vs[e.source]
                usrid = acg.get_usrid(vu['name'])
                weight = e['weight']
                requesters_usrids.append(usrid)
                weights.append(weight)

                #Values of vu sorted according to rel_attnames
                usrvalues = []
                for attname in vu.attribute_names():
                    if not attname in ['type','name']:
                        usrvalues.append(vu[attname])

                #Collect the usrvalues of all the requesters of vr
                usrvalues_list.append(usrvalues)


            #Create the list of value entries by repeating the elements
            #of usrvalues_list according to the weigths
            valsentries = []
            entidx_to_usrid = [] #It maps the entry idx to the corresponding requester usrid
            for i,w in enumerate(weights):
                usrid = requesters_usrids[i]
                usrvalues = usrvalues_list[i]
                for _ in range(w):
                    valsentries.append(usrvalues_list[i])
                    entidx_to_usrid.append(usrid)
            #Include in output dictionary
            resid_to_valsentries[resid] = valsentries


            #Calculate the local support
            local_s = global_f/len(valsentries)
            run_flag = True
            if local_s > 1.0:
                local_s = 1.0
                run_flag = False
            if local_s < 0.01:
                local_s = 0.01

            if verbose:
                print(local_s)

            #Transform the value format to tuple furmat
            entries_t = ut.to_tuple_format(valsentries)
            #Compute the a-v patterns with closed frequent itemsets
            freq_itemsets,_ = apriori(entries_t, min_support=local_s, min_confidence=1.0)
            freq_itemsets = _filter_closed(freq_itemsets) #keep those that are closed

            #Collect a-v patterns for this resid
            patterns = []
            for L in lengths:
                if L in freq_itemsets.keys():
                    #Get patterns
                    for f_itemset,freq in freq_itemsets[L].items():
                        pattern = list(f_itemset) #Unpack tuple of tuples
                        patterns.append(pattern)

            if run_flag == False:
                patterns = []

            resid_to_patterns[resid] = patterns #Include in output dictionary


    return resid_to_valsentries, resid_to_patterns
