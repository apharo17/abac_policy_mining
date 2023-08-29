import pandas as pd
import numpy as np
import igraph as ig

#This modules contains the functions to load and manipute the access control
#graphs from two datasets (amazon kaggle y amazon uci)

def load_azuci_graph(rel_attnames = None, isWeighted=True, last_eidx = -1):


    df_history = pd.read_csv('../data/access_log.csv')

    if last_eidx > 0:
        df_history = df_history[df_history.index.isin(range(last_eidx+1))]

    #For azuci the entries turn out to be a history due to the time component
    df_users = pd.read_csv('../data/az_uci_users.csv')

    #Delete 'Unnamed: 0' column and set PERSON_ID' at the beginning
    df_users = df_users[['PERSON_ID', 'PERSON_BUSINESS_TITLE', 'PERSON_BUSINESS_TITLE_DETAIL',
    'PERSON_COMPANY', 'PERSON_DEPTNAME', 'PERSON_JOB_CODE',
    'PERSON_JOB_FAMILY', 'PERSON_LOCATION', 'PERSON_MGR_ID', 'PERSON_ROLLUP_1',
    'PERSON_ROLLUP_2', 'PERSON_ROLLUP_3']]

    if rel_attnames == None:
        rel_attnames = list(df_users.columns)[1:]

    df_pos_history = df_history[df_history['ACTION']=='add_access']
    df_neg_history = df_history[df_history['ACTION']=='remove_access']

    #---Create the mapping between a usrid and its index in df_users
    uid_col_list = df_users['PERSON_ID'].values.tolist()
    uid_to_idx = {uid:idx for idx,uid in enumerate(uid_col_list)}

    #---Extract relevant attributes from df_users---
    df_info = df_users[rel_attnames]
    idx_to_info = df_info.values.tolist()

    #Create the mapping between usrids and resids using the df_history (just positives)
    usrids = list(df_pos_history['LOGIN'].unique()) #It does not matter the order
    resids = list(df_pos_history['TARGET_NAME'].unique()) #It does not matter the order
    atts_keys = rel_attnames

    #Extracting the duplicates from the positives to calculate the weights
    dfgb = df_pos_history.groupby(['LOGIN','TARGET_NAME'])
    df_temp = dfgb.size().to_frame(name = 'COUNT').reset_index()

    usrid_to_resid = df_temp[['LOGIN','TARGET_NAME']].values.tolist() #This instance has no duplicates
    weights = df_temp['COUNT'].values.tolist()

    #TODO: extract negatives for validation

    #-----Create graph-----
    #Create types list
    types = []
    for i in range(len(usrids)):
        types.append(False)
    for i in range(len(resids)):
        types.append(True)

    #Create edgeslist
    #gidx is the vertex index in the graph, according the vertex was appended
    #in the types list
    usrid_to_gidx = {usrid:i for i,usrid in enumerate(usrids)}
    resid_to_gidx = {resid:i+len(usrids) for i,resid in enumerate(resids)}
    edgelist = []
    for row in usrid_to_resid:
        usrid = row[0]
        resid = row[1]
        edgelist.append((usrid_to_gidx[usrid],resid_to_gidx[resid]))

    #Create bipartite graph
    gur = ig.Graph.Bipartite(types=types, edges=edgelist)

    #Assign labels and attributes to the vertices
    for i,usrid in enumerate(usrids):
        usrlabel = 'USRID_' + str(usrid)
        gur.vs[i]['name'] = usrlabel
        atts_vals = idx_to_info[uid_to_idx[usrid]]
        for j,attname in enumerate(atts_keys):
            gur.vs[i][attname] = atts_vals[j]
    for i,resid in enumerate(resids):
        reslabel = 'RESID_' + str(resid)
        gur.vs[i+len(usrids)]['name'] = reslabel

    #Assign weights
    if isWeighted:
        gur.es['weight'] = weights

    return gur



def load_azkag_graph(rel_attnames=None, isWeighted=True):

    df_emp_access = pd.read_csv('../data/train.csv')
    attnames_list = list(df_emp_access.columns)[2:]

    #---Group by the column header: attvalues, residx, resolution---
    dfgb_temp = df_emp_access.groupby(attnames_list + ['RESOURCE','ACTION'])
    #convert DataFrameGroupBy to df (I dont know how it works, but it uses a count field)
    df_temp = dfgb_temp.size().to_frame(name = 'count').reset_index()
    #check the file if needed
    #df_temp.to_csv('temp_header.csv') #22 02 14 All the counts must be equal to 1
    #drop count column
    df_temp = df_temp.drop('count', axis=1)

    #---Add user indexes---
    dfgb_temp = df_temp.groupby(attnames_list)

    dfs_list = []
    usridx = 0
    for _,df in dfgb_temp:
        k = df.shape[0]
        df.insert(loc=0, column='USR_ID', value=[usridx for _ in range(k)])
        dfs_list.append(df)
        usridx += 1
    df_temp = pd.concat(dfs_list)
    #22 02 14 Exporting df_temp to csv, creates temp_users.csv

    #---Create dataframe for users and entries---
    df_users = df_temp[['USR_ID']+attnames_list].drop_duplicates().reset_index()
    df_users = df_users.drop('index', axis=1) #improve this line
    df_entries = df_temp[['USR_ID', 'RESOURCE', 'ACTION']]



    #Codigo nuevo#################################
    if rel_attnames == None:
        rel_attnames = list(df_users.columns)

    #---Split entries in positive and negative samples
    df_pos_entries = df_entries[df_entries['ACTION']==1]
    df_neg_entries = df_entries[df_entries['ACTION']==0]

    #---Create the mapping between a usrid and its index in df_users
    uid_col_list = df_users['USR_ID'].values.tolist()
    uid_to_idx = {uid:idx for idx,uid in enumerate(uid_col_list)}

    #---Extract relevant attributes from df_users---
    df_info = df_users[rel_attnames]
    idx_to_info = df_info.values.tolist()

    #Create the mapping between usrids and resids using the df_pos_entries
    usrids = list(df_neg_entries['USR_ID'].unique()) #It does not matter the order
    resids = list(df_neg_entries['RESOURCE'].unique()) #It does not matter the order
    atts_keys = rel_attnames

    #Extracting the duplicates from the positives to calculate the weights
    dfgb = df_neg_entries.groupby(['USR_ID','RESOURCE'])
    df_temp = dfgb.size().to_frame(name = 'COUNT').reset_index()

    usrid_to_resid = df_temp[['USR_ID','RESOURCE']].values.tolist() #This instance has no duplicates
    weights = df_temp['COUNT'].values.tolist()

    #TODO: extract negatives for validation


    #-----Create graph-----
    #Create types list
    types = []
    for i in range(len(usrids)):
        types.append(False)
    for i in range(len(resids)):
        types.append(True)

    #Create edgeslist
    #gidx is the vertex index in the graph, according the vertex was appended
    #in the types list
    usrid_to_gidx = {usrid:i for i,usrid in enumerate(usrids)}
    resid_to_gidx = {resid:i+len(usrids) for i,resid in enumerate(resids)}
    edgelist = []
    for row in usrid_to_resid:
        usrid = row[0]
        resid = row[1]
        edgelist.append((usrid_to_gidx[usrid],resid_to_gidx[resid]))

    #Create bipartite graph
    gur = ig.Graph.Bipartite(types=types, edges=edgelist)

    #Assign labels and attributes to the vertices
    for i,usrid in enumerate(usrids):
        usrlabel = 'USRID_' + str(usrid)
        gur.vs[i]['name'] = usrlabel
        atts_vals = idx_to_info[uid_to_idx[usrid]]
        for j,attname in enumerate(atts_keys):
            gur.vs[i][attname] = atts_vals[j]
    for i,resid in enumerate(resids):
        reslabel = 'RESID_' + str(resid)
        gur.vs[i+len(usrids)]['name'] = reslabel

    #Assign weights
    if isWeighted:
        gur.es['weight'] = weights

    return gur



def get_labels(gur, byweights=False):
    #It returs the list of user and resource labels of G
    #If byweights is False, the lists are decreasing sorted by vertex degree
    #If byweights is True, the lists are decreasing sorted by sum of the weights
    #of the incident edges

    usrlabel_to_score = dict()
    reslabel_to_score = dict()

    if not byweights:
        for v in gur.vs:
            degree = len(gur.neighbors(v, mode='all'))
            if v['type'] == False:
                usrlabel_to_score[v['name']] = degree
            else:
                reslabel_to_score[v['name']] = degree
    else:
        for v in gur.vs:
            #https://stackoverflow.com/questions/30660808/how-to-get-the-vertices-from-an-edge-using-igraph-in-python
            weight = 0
            for eidx in gur.incident(v, mode='all'):
                weight += gur.es[eidx]['weight']
            if v['type'] == False:
                usrlabel_to_score[v['name']] = weight
            else:
                reslabel_to_score[v['name']] = weight

    temp = sorted(usrlabel_to_score.items(), key=lambda item: item[1], reverse=True)
    usrlabels = [tup[0] for tup in temp]
    temp = sorted(reslabel_to_score.items(), key=lambda item: item[1], reverse=True)
    reslabels = [tup[0] for tup in temp]

    return usrlabels, reslabels



def get_usrattvalues(gur, usrids, attnames):
    #It gets the user attribute values of each usrid in usrids
    #usrids is a subset list of the user ids of gur
    #attnames is the list of user attribute names of gur
    #The list of lists of attributes values is returned in the order of usrids
    #The order of values is according to the attnames order

    attvals_list = []

    for usrid in usrids:
        usrlabel = 'USRID_'+str(usrid)
        v = gur.vs.find(usrlabel)
        attvals = []
        for attname in attnames:
            attvals.append(v[attname])
        attvals_list.append(attvals)

    return attvals_list


def get_usrlabel(usrid):
    return 'USRID_'+str(usrid)


def get_reslabel(resid):
    return 'RESID_'+str(resid)


def get_usrlabels(usrids):
    usrlabels = []
    for usrid in usrids:
        usrlabels.append(get_usrlabel(usrid))
    return usrlabels


def get_reslabels(resids):
    reslabels = []
    for resid in resids:
        reslabels.append(get_reslabel(resid))
    return reslabels


def get_usrid(usrlabel):
    return int(usrlabel[6:])


def get_resid(reslabel):
    return int(reslabel[6:])


def get_usrids(usrlabels):
    usrids = []
    for usrlabel in usrlabels:
        usrids.append(get_usrid(usrlabel))
    return usrids


def get_resids(reslabels):
    resids = []
    for reslabel in reslabels:
        resids.append(get_resid(reslabel))
    return resids


def get_largestcc(gur):
    #It returns the largest connected component of gur
    ccidx_to_vertexidxs = [cc for cc in gur.clusters()]
    ccidx_to_size = [len(cc) for cc in ccidx_to_vertexidxs]
    max_index = ccidx_to_size.index(max(ccidx_to_size))
    maxcc_gur = gur.induced_subgraph(ccidx_to_vertexidxs[max_index])
    return maxcc_gur
