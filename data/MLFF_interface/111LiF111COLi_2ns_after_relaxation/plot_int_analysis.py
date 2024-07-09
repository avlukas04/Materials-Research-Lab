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

interface=read('POSCAR')
plotCN = False
plotANG = True
slice_width = 6

#----Select bond CN to plot
CNtoplot = [["Li","O"]]
ANGtoplot = [('O','Li','C'),('Li','C','Li'),('C','O','Li'),('O','Li','O')]
ANGtoplot = [('O','Li','O'),('O','Li','C'),('C','O','Li'),('Li','O','Li'),('Li','C','Li'),('C','Li','C')]

#----Select saved dictionaries to load

#structlist = ['6w_CNdictlist.json','../101LF001COLi_2ns_after_relaxation/6w_CNdictlist.json']
structlist = ['6w_ANGdictlist.json']
#structlist = ['3w_ANGdictlist.json']
structload = []

for s in structlist:
    with open(s) as user_file:
        structload.append(json.load(user_file))

num_slices = int(LA.norm(interface.get_cell()[2])/slice_width)
xrange = np.linspace(0,LA.norm(interface.get_cell()[2]),num_slices)

def atms2str(tup):
    stringsave = ""
    for i in tup:
        stringsave=stringsave+i
    return stringsave

if plotCN:
    CN=0

    for sID, struc in enumerate(structload):
        atmstr = atms2str(CNtoplot[CN])
        listoave = []

        for s in struc:
            try:
                listoave.append(s[atmstr]['ave'])
            except: listoave.append(0)

        xrange = np.linspace(0,LA.norm(interface.get_cell()[2]),num_slices)
        plt.plot(xrange, listoave,'o-',label=str(structlist[sID]))
        plt.xlabel("Distance along norm of interface (Ang)")
        plt.ylabel("Coordination Number bonds "+atmstr)
        plt.legend(loc="upper center",bbox_to_anchor=(0.5, 1.15))
    plt.savefig(atmstr+"_slice"+str(slice_width)+"_CN.png")

    plt.show()

if plotANG:

    for sID, struc in enumerate(structload):
        for ang in ANGtoplot:
            atmstr = atms2str(ang)
            listoave = []
            for slice in struc:
                try:
                    listoave.append(slice[atmstr][0])
                except: listoave.append(0)
            plt.plot(xrange, listoave,'o-',label=str(atmstr))
            plt.xlabel("Distance along norm of interface (Ang)")
            plt.ylabel("Counts of Atom with Angles" )

        #plt.title(str(struc))
        plt.legend(loc="upper left")
        plt.savefig("struct"+str(sID)+"slice"+str(slice_width)+"_ANG.png")
        plt.show()
        #
        
        #plt.show()
