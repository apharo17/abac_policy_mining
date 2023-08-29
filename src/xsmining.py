import igraph as ig
from efficient_apriori import apriori
from src.biclique import Biclique
from src import bcgraph as bcg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from collections import Counter
import math

def _load_system(name):
    #It loads users and resources, and the corresponding attribute values
    #of the system under the name 'name'
    #The valid values of 'name' are: 'HC', 'PM' and 'UN'

    usrlabel_to_attvals = dict()
    usrname_to_usrlabel = dict()
    reslabel_to_attvals = dict()
    resname_to_reslabel = dict()

    usrcount = 1
    rescount = 1

    fname = 'healthcare_10_5.abac'
    if name == 'PM':
        fname = 'projectmanagement_10_9.abac'
    elif name == 'UN':
        fname = 'university_6_9.abac'

    f = open(fname, 'r')

    for line in f.readlines():
        arr = line.split('(')

        if arr[0] == 'userAttrib':
            info = arr[1].strip(')\n')

            attvals = []
            usrid = ''
            arr2 = info.split(',')
            for i,x in enumerate(arr2):
                attval = x.strip()
                if i == 0:
                    usrname = attval
                    arr3 = attval.split('-')
                    val = arr3[0][:-1]
                    if name == 'HC':
                        attvals.append(('type', val))
                    elif name == 'PM':
                        if val.startswith('ldr'):
                            attvals.append(('type', 'ldr'))
                        elif val.startswith('nonemployee'):
                            attvals.append(('type', 'nonemployee'))
                        elif val.startswith('employee'):
                            attvals.append(('type', 'employee'))
                        elif val.startswith('manager'):
                            attvals.append(('type', 'manager'))
                        elif val.startswith('acc'):
                            attvals.append(('type', 'acc'))
                        elif val.startswith('aud'):
                            attvals.append(('type', 'aud'))
                        elif val.startswith('planner'):
                            attvals.append(('type', 'planner'))
                        else:
                            attvals.append(('type', val))
                    elif name == 'UN':
                        attvals.append(('type', val))

                else:
                    arr3 = attval.split('=')
                    attvals.append((arr3[0], arr3[1]))

            usrlabel = 'USRID_' + str(usrcount)
            usrlabel_to_attvals[usrlabel] = attvals
            usrname_to_usrlabel[usrname] = usrlabel
            usrcount += 1

        elif arr[0] == 'resourceAttrib':

            info = arr[1].strip(')\n')

            attvals = []
            resid = ''
            arr2 = info.split(',')
            for i,x in enumerate(arr2):
                attval = x.strip()
                if i == 0:
                    resname = attval
                else:
                    arr3 = attval.split('=')
                    attvals.append((arr3[0], arr3[1]))

            reslabel = 'RESID_' + str(rescount)
            reslabel_to_attvals[reslabel] = attvals
            resname_to_reslabel[resname] = reslabel
            rescount += 1

    f.close()

    return usrlabel_to_attvals, usrname_to_usrlabel, \
    reslabel_to_attvals, resname_to_reslabel


def _load_log(name, usrname_to_usrlabel, resname_to_reslabel):
    #It loads the log entries of the system under the name 'name'
    #The valid values of 'name' are: 'HC', 'PM' and 'UN'

    usrlabel_to_reslabel = []
    usrlabels = set()
    reslabels = set()

    fname = 'healthcare_10_5.log'
    if name == 'PM':
        fname = 'projectmanagement_10_9.log'
    elif name == 'UN':
        fname = 'university_6_9.log'

    f = open(fname, 'r')

    for line in f.readlines():
        arr = line.split(',')
        if len(arr) > 1:
            usrlabel = usrname_to_usrlabel[arr[1]]
            reslabel = resname_to_reslabel[arr[2]]
            usrlabels.add(usrlabel)
            reslabels.add(reslabel)
            usrlabel_to_reslabel.append((usrlabel, reslabel))

    f.close()

    return list(usrlabels), list(reslabels), usrlabel_to_reslabel


#-------------------------------------------------------------------------------
def load_dataset(name):
    #It loads the dataset of the system under the name 'name'
    #The valid values of 'name' are: 'HC', 'PM' and 'UN'

    usrlabel_to_attvals, usrname_to_usrlabel, \
    reslabel_to_attvals, resname_to_reslabel = _load_system(name)
    usrlabels, reslabels, usrlabel_to_reslabel = _load_log(name, usrname_to_usrlabel, resname_to_reslabel)

    return usrlabels, usrlabel_to_attvals, usrname_to_usrlabel, \
    reslabels, reslabel_to_attvals, resname_to_reslabel,usrlabel_to_reslabel


def load_gur(usrlabels, reslabels, usrlabel_to_reslabel, save=False):

    #It returns the access control graph from the system information
    #If 'save' is True: it creates the gml file and the files for BCFinder

    types = []
    for i in range(len(usrlabels)):
        types.append(False)
    for i in range(len(reslabels)):
        types.append(True)

    usrlabels_to_vidx = dict()
    for i,usrlabel in enumerate(usrlabels):
        usrlabels_to_vidx[usrlabel] = i

    reslabels_to_vidx = dict()
    for i,reslabel in enumerate(reslabels):
        reslabels_to_vidx[reslabel] = i+len(usrlabels)

    edgelist = []
    for row in usrlabel_to_reslabel:
        usrlabel = row[0]
        reslabel = row[1]
        edgelist.append((usrlabels_to_vidx[usrlabel],reslabels_to_vidx[reslabel]))

    gur = ig.Graph.Bipartite(types=types, edges=edgelist)

    for i,usrlabel in enumerate(usrlabels):
        gur.vs[i]['name'] = usrlabel

    for i,reslabel in enumerate(reslabels):
        gur.vs[i+len(usrlabels)]['name'] = reslabel

    #Create files
    if save:

        #Gml graph
        gur.write_gml('gur.gml')

        #BCFinder edges
        f = open('lehmann_input/edges.txt', 'w')
        for edge in gur.es:
            usridx = edge.source
            residx = edge.target
            usrid = int(gur.vs[usridx]['name'][6:])
            resid = int(gur.vs[residx]['name'][6:])
            stroutput = "{0} {1}\n".format(usrid,resid)
            f.write(stroutput)
        f.close()

        #BCFinder user labels
        f = open('lehmann_input/users.txt', 'w')
        for v in gur.vs:
            if v['type'] == False:
                usr_label = v['name']
                usrid = int(usr_label[6:])
                stroutput = "{0} {1}\n".format(usrid,usr_label)
                f.write(stroutput)
        f.close()

        #BCFinder resource get_labels
        f = open('lehmann_input/resources.txt', 'w')
        for v in gur.vs:
            if v['type'] == True:
                res_label = v['name']
                resid = int(res_label[6:])
                stroutput = "{0} {1}\n".format(resid,res_label)
                f.write(stroutput)
        f.close()

    return gur


#-------------------------------------------------------------------------------
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


def _check(pattern):
    count_u = 0
    count_r = 0
    for attval in pattern:
        if attval[0] == 'U':
            count_u += 1
        if attval[0] == 'R':
            count_r += 1

    if count_u > 0 and count_r > 0:
        return True
    return False


def compute_avprules(minsup, usrlabel_to_attvals, reslabel_to_attvals, usrlabel_to_reslabel):

    #It returns the rules from avpatterns given a min support

    entries_t = []

    for tup in usrlabel_to_reslabel:
        usrlabel = tup[0]
        reslabel = tup[1]
        usrattvals = usrlabel_to_attvals[usrlabel]
        resattvals = reslabel_to_attvals[reslabel]
        usrattvals = [('U',attval) for attval in usrattvals]
        resattvals = [('R',attval) for attval in resattvals]
        entries_t.append(usrattvals+resattvals)

    freq_itemsets,_ = apriori(entries_t, min_support=minsup, min_confidence=1.0)
    freq_itemsets = _filter_closed(freq_itemsets)

    avpatterns = []
    for L in freq_itemsets.keys():
        if L > 1:
            #Get patterns
            for f_itemset,freq in freq_itemsets[L].items():
                avpattern = list(f_itemset) #Unpack tuple of tuples
                if _check(avpattern):
                    avpatterns.append(avpattern)

    rules = []
    for avpattern in avpatterns:
        usrattvals = set()
        resattvals = set()
        for attval in avpattern:
            if attval[0] == 'U':
                usrattvals.add(attval[1])
            elif attval[0] == 'R':
                resattvals.add(attval[1])
        rules.append([usrattvals, resattvals])

    return rules


#-------------------------------------------------------------------------------
def logcov(rules, usrlabel_to_reslabel, usrlabel_to_attvals, reslabel_to_attvals):

    #Given a set of rules and a set of log entries 'usrlabel_to_reslabel',
    #it computes the log coverage

    count = 0
    for tup in usrlabel_to_reslabel:
        usrlabel = tup[0]
        reslabel = tup[1]
        usrattvals = usrlabel_to_attvals[usrlabel]
        resattvals = reslabel_to_attvals[reslabel]

        for rule in rules:
            if len(set(usrattvals)&rule[0]) == len(rule[0]):
                if len(set(resattvals)&rule[1]) == len(rule[1]):
                    count += 1
                    break

    return count/len(usrlabel_to_reslabel)


def rescov(rules, gur, reslabel_to_attvals):
    #Given a set of rules and the acg 'gur',
    #it computes the resource coverage

    count = 0
    rescount = 0
    for v in gur.vs:
        if v['type'] == True:
            rescount += 1
            resattvals = reslabel_to_attvals[v['name']]
            for rule in rules:
                if len(set(resattvals)&rule[1]) == len(rule[1]):
                    count += 1
                    break

    return count/rescount


#-------------------------------------------------------------------------------
def _compute_pattern(gur, usrids, resids, usrlabel_to_attvals, reslabel_to_attvals):

    entries_t = []

    for usrid in usrids:
        usrlabel = 'USRID_' + str(usrid)
        usrattvals = usrlabel_to_attvals[usrlabel]
        usrattvals = [('U',attval) for attval in usrattvals]
        for resid in resids:
            reslabel = 'RESID_' + str(resid)
            resattvals = reslabel_to_attvals[reslabel]
            resattvals = [('R',attval) for attval in resattvals]
            entries_t.append(usrattvals+resattvals)

    #The pattern must have 1.0 of support
    freq_itemsets,_ = apriori(entries_t, min_support=1.0, min_confidence=1.0)

    lengths = freq_itemsets.keys()

    if len(lengths) > 0: #freq_itemsets not empty
        maxlen = max(lengths)
        if maxlen >= 1:
            if len(freq_itemsets[maxlen].keys()) == 1: #It has a unique pattern
                #Get pattern
                temp = list(freq_itemsets[maxlen].keys())[0]
                pattern = list(temp) #Unpack tuple of tuples
                return pattern

    return None


def load_bicliques(name, gur, k, usrlabel_to_attvals, reslabel_to_attvals):

    #It returns list of bicliques
    fname = 'bcfinder_bicliques/healthcare_edges.bi'
    if name == 'PM':
        fname = 'bcfinder_bicliques/project_edges.bi'
    elif name == 'UN':
        fname = 'bcfinder_bicliques/university_edges.bi'

    id = 0
    bcs = []

    usrids = []
    resids = []
    f = open(fname, 'r')
    count = 0
    for line in f.readlines():
        arr = line.split(' ')
        arr = arr[:-1] #because it has a space at the end

        if count == 0: #users
            usrids = [int(s) for s in arr]
            count += 1
        elif count == 1: #resources
            resids = [int(s) for s in arr]
            count += 1
            #Create biclique
            if len(usrids) >= k[0] and len(resids) >= k[1]:
                bc = Biclique(id=id, usrids=usrids, resids=resids)
                pattern = _compute_pattern(gur, usrids, resids, usrlabel_to_attvals, reslabel_to_attvals)
                bc.set_pattern(pattern)
                bcs.append(bc)
                #print(id) #For debug
                id += 1
        else:
            count = 0

    f.close()

    return bcs


def plot_sizes_freq(bcs,fname):
    #It shows a colorimetric plot of the frequency sizes of bcs,
    #and saves the image in fname
    sizes = []

    for bc in bcs:
        sizes.append((len(bc.get_usrids()),len(bc.get_resids())))
    sizes_counter = Counter(sizes)
    dict_temp = dict(sizes_counter)
    #Ascending order in freq to avoid overlapping of frequent sizes
    dict_temp = {k: v for k, v in sorted(dict_temp.items(), key=lambda item: item[1])}

    list_temp = list(dict_temp.keys())
    unzipped_object = zip(*list_temp)
    unzipped_list = list(unzipped_object)

    x = unzipped_list[0]
    y = unzipped_list[1]
    f = list(dict_temp.values())

    plt.scatter(x,y,c=f,marker='.', norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Number of users')
    plt.ylabel('Number of resources')
    #plt.savefig(fname+'.png')
    plt.show()


def save_sizedistrib(bcs):

    sizes = []
    for bc in bcs:
        sizes.append((len(bc.get_usrids()),len(bc.get_resids())))
    sizes_counter = Counter(sizes)
    dict_temp = dict(sizes_counter)
    #Ascending order in freq to avoid overlapping of frequent sizes
    dict_temp = {k: v for k, v in sorted(dict_temp.items(), key=lambda item: item[1])}

    list_temp = list(dict_temp.keys())
    unzipped_object = zip(*list_temp)
    unzipped_list = list(unzipped_object)

    freqs = list(dict_temp.values())
    sumf = sum(freqs)
    strout = 'k1 k2 P\n'
    f = open('biclique_distrib.dat','w')
    f.write(strout)
    for i in range(len(dict_temp)):
        k1 = unzipped_list[0][i]*1.0
        k2 = unzipped_list[1][i]*1.0
        freq = freqs[i]*1.0
        prob = freq/sumf
        prob=math.log10(prob)
        strout = str(k1)+' '+str(k2)+' '+str(prob)+'\n'
        f.write(strout)

    f.close()


def get_subbcs(bcs):
    subbcs = []
    for bc in bcs:
        pattern = bc.get_pattern()
        if pattern != None and _check(pattern):
            subbcs.append(bc)
    return subbcs


def get_num_covered_edges(bcs):
    #Return the number of edges of gur that is covered by bcs
    edges_set = set()
    for bc in bcs:
        for usrid in bc.get_usrids():
            for resid in bc.get_resids():
                edges_set.add((usrid,resid))
    return len(edges_set)


def get_num_covered_users(bcs):
    usrids_set = set()

    for bc in bcs:
        usrids_ = bc.get_usrids()

        if len(usrids_set) == 0: usrids_set = set(usrids_)
        else: usrids_set |= set(usrids_)

    return len(usrids_set)


def get_num_covered_resources(bcs):
    resids_set = set()

    for bc in bcs:
        resids_ = bc.get_resids()

        if len(resids_set) == 0: resids_set = set(resids_)
        else: resids_set |= set(resids_)

    return len(resids_set)


def load_bcgraph(gur, subbcs, save=False):
    th = 0.01
    W = bcg.get_weight_matrix(gur, subbcs, verbose=False)
    bcgraph = bcg.create_bcgraph(subbcs, W, th)

    if save:
        bcgraph.write_gml('bcgraph.gml')

    return bcgraph


def compute_gprules(bcgraph, subbcs):
    bcid_to_bc = {bc.get_id():bc for bc in subbcs}
    gps = bcg.dfs(bcgraph, bcid_to_bc, sa=1)
    rules = []
    bcs_list = []
    for gp in gps:
        usrattvals = set()
        resattvals = set()
        for attval in gp[0]:
            if attval[0] == 'U':
                usrattvals.add(attval[1])
            elif attval[0] == 'R':
                resattvals.add(attval[1])
        rules.append([usrattvals, resattvals])
        bcs_list.append(gp[1])
    return rules, bcs_list
