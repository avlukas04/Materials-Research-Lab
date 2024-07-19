#!/usr/bin/env python

from ase import Atoms
from ase.io import read,write
from ase.geometry.analysis import Analysis
import numpy as np
import itertools
import matplotlib.pyplot as plt
from ase.build import make_supercell
from numpy import linalg as LA
import os
import statistics
from ase.build.tools import cut
from ase.visualize import view
import json
# -- Local imports
#from .npencoder import NpEncoder

def main():

    interface=read('POSCAR')
    print("Analyzing ",interface)

    slice_width = 3

    #Set up pairs and triplets of atoms to check for neighbors.
    pairs = [('C', 'F'), ('C', 'Li'), ('C', 'O'), ('F', 'Li'), ('F', 'O'), ('Li', 'O')]
    symbs = ['C', 'F', 'Li', 'O']
    triples = ['C', 'F', 'Li', 'O','C', 'F', 'Li', 'O','C', 'F', 'Li', 'O']
    all_anglesID = np.unique(np.asarray(list(itertools.permutations(triples,3))),axis=0)

    #Make a list of all possible angles triplets
    all_anglesID = np.unique(np.asarray(list(itertools.permutations(triples,3))),axis=0)

    num_slices = int(LA.norm(interface.get_cell()[2])/slice_width)

    #calculate angle and bond dictionaries and save
    CN_int_list = []
    ANG_int_list = []
    for i in np.linspace(0,1,num_slices):
        slice_int = cut(interface, (1,0,0), (0,1,0),(0,0,1), clength=slice_width,origo= (0,0,i))
        slice_int.set_pbc([ True,  True,  False])
        CN_int_list.append(dictCN(slice_int))
        ANG_int_list.append(dictAng(slice_int,all_anglesID))


    with open(str(slice_width)+'w_CNdictlist.json', 'w') as f:
        #json.dump(CN_int_list, f)
        json.dump(CN_int_list, f, indent=4, cls=NpEncoder)

    with open(str(slice_width)+'w_ANGdictlist.json', 'w') as f:
        json.dump(ANG_int_list, f, indent=4, cls=NpEncoder)
        #json.dump(ANG_int_list, f)

def atms2str(tup):
    stringsave = ""
    for i in tup:
        stringsave=stringsave+i
    return stringsave


def getCN(bondlist):
    bondlst = np.asarray(bondlist[0])
    atm_cntrs = np.unique(bondlst[:,0])
    cnList = []
    for i in atm_cntrs:
        cn_bool = bondlst[:,0]==i
        cnList.append(cn_bool.sum())
    return np.average(cnList),cnList

def dictCN(struct):
    #print(struct.get_chemical_formula(mode="hill",empirical=True))
    ana = Analysis(struct)
    symbs = struct.get_chemical_symbols()
    ### Count bond numbers - put in dictionary
    CNdict = {}
    for p in itertools.combinations(np.unique(symbs),2):
        tup=atms2str((p[0],p[1]))
        bondlist = ana.get_bonds(p[0], p[1], unique=True)
        if len(bondlist[0])>0:
            ave,CNlist = getCN(bondlist)
            CNdict[tup] = {'ave':ave,'CNlist':CNlist}
        else:
            CNdict[tup] = {'ave':0,'CNlist':[]}

    for s in np.unique(symbs):
        bondlist = ana.get_bonds(s, s, unique=False)
        tup=atms2str((s,s))
        if len(bondlist[0])>0:
            ave,CNlist = getCN(bondlist)
            CNdict[tup] = {'ave':ave,'CNlist':CNlist}
        else:
            CNdict[tup] = {'ave':0,'CNlist':[]}

    return CNdict

def getAN(anglelist):
    anglst = np.asarray(anglelist[0])
    cntrs, counts = np.unique(anglst[:,1],return_counts=True)
    return counts, len(cntrs), np.average(counts)

def dictAng(struct,all_anglesID):
    #Populate angle dictionary with empty lists
    ANGdict={}
    for p in all_anglesID:
        tup=atms2str(p)
        ANGdict[tup]=[]

    ana = Analysis(struct)
    symbs = struct.get_chemical_symbols()

    for p in all_anglesID:
        anglelist = ana.get_angles(p[0], p[1],p[2], unique=True)
        tup=atms2str(p)
        if len(anglelist[0])>0:
            counts, num, ave = getAN(anglelist)
            #ANGdict[tuple(p)]=ave
            ANGdict[tup].append(ave)
        else: ANGdict[tup].append(0)
            
    return ANGdict

class NpEncoder(json.JSONEncoder):
    """
    TODO: Add documentation string describing usage of class.
    """

    def default(self, obj):
        """
        TODO: Add documentation string for class method.
        """
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()

