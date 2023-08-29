import numpy as np


def to_tuple_format(entries):
    #Transforms each entry in entries to a list of tuples (att_idx, att_val)

    t_list = []
    num_atts = len(entries[0])

    for entry in entries:
        t = []
        for att_idx in range(num_atts):
            att_val = entry[att_idx]
            t.append((att_idx,att_val))
        t_list.append(tuple(t))

    return t_list


def _overlap_coeff(pattset1, pattset2):
    if len(pattset1) < len(pattset2):
        minlen = len(pattset1)
    else:
        minlen = len(pattset2)

    return len(pattset1&pattset2)/minlen


def filter_closed(freq_itemsets):
    #Given a dictionary of dictionaries freq_itemsets,
    #whose keys are sizes: 1, 2, 3,..., l
    #and the values are dictionaries: {itemset1:sup1, itemset2:sup2...}
    #This function filters the closed frequent itemsets

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
