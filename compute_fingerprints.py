import copy
import os
import ruamel.yaml as yaml
import json
from glob import glob
import numpy as np
from mp_api.client import MPRester
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.structure.sites import PartialsSiteStatsFingerprint
import pymatgen.core
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.local_env import site_is_of_motif_type
from sklearn.decomposition import PCA
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from pymatgen.core.periodic_table import Element

def main():
    ### ---- Load in structures
    #directory = "/Users/alukas/Desktop/Materials-Research-Lab/data/mixtures"
    directory = "/Users/alukas/Desktop/Materials-Research-Lab/data/mixtures"
    fingprints = []
    fingprintschem = []
    structures = []
    filenames,structnames = loaddata(directory)
    test1 = True
    sites = False
    for filename in filenames:
       #print(structnames)
       print(filename)
       structure = Structure.from_file(filename)
       if test1:
           #filename = "/Users/alukas/Desktop/Materials-Research-Lab/data/MLFF_interface/int_slice_3layers.POSCAR.\"
           structure = Structure.from_file(filename)
           #print(filename)
           #fingprints.append(calc_cnn(structure,0))
           ### ---- Test just one structure, to run quickly
           fingprints.append(calc_ssf(structure))
           #filename = "/Users/alukas/Desktop/Materials-Research-Lab/data/MLFF_interface/int_slice_1layer.POSCAR.vasp"
           structure = Structure.from_file(filename)
           fingprints.append(calc_ssf(structure))
           #fingprints.append(calc_ssfp(structure))
           break
       if sites:
           op_types = load_ops()
           cnfp = CrystalNNFingerprint(op_types, distance_cutoffs=None, x_diff_weight=0)
           for idx,s in enumerate(structure.sites):
               #if idx in [list of atoms at interface]
               if str(s.specie.symbol)=="Li":
                   sitefingdict = {"Li":[],"O":[],"C":[],"F":[]}
                   sitefp = []
                   sitefingdict["Li"].append(calc_cnn(structure,idx))
                   #fingprints.append(calc_cnn(structure,idx))
                   nndata = cnfp.cnn.get_nn_data(structure, idx)
                   cn = 12
                   #cn = cnfp.cnn.get_cn(structure,idx)
                   neigh_sites = [d["site_index"] for d in nndata.cn_nninfo[cn]]
                   for neigh in neigh_sites:
                       symb = str(structure.sites[neigh].specie.symbol)
                       sitefingdict[symb].append(calc_cnn(structure,neigh))
                   for i in ["Li","O","C","F"]:
                       mean = np.mean(np.asarray(sitefingdict[i]),axis=0)
                       if np.isnan(mean).any():
                           sitefp.append(np.zeros(61))
                       else:
                           sitefp.append(mean)
                   localfingerprint = np.hstack(sitefp)        #need to contatenate, not apppend
                   fingprints.append(localfingerprint)

                  #print(sitefingprint)
           #print(np.asarray(fingprints).shape)
           break
       #fingprints.append(calc_ssf(structure))
       fingprints.append(calc_ssfp(structure))
       fingprintschem.append(calc_ssfp(structure,chem=True))
       

    #np.save("fingerprints/pfingerprints_all_chem.npy",np.asarray(fingprintschem))
    #np.save("fingerprints/fingerprints_int.npy",np.asarray(fingprints))
    
    # Ensure the directory exists
    print(fingprints)
    output_dir = "fingerprints"
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "fingerprints_int.npy"), np.asarray(fingprints))

    ### ---- Load in fingerprints if already computed
    #fingprints = np.load("fingerprints_all.npy")

    ### ---- Load in VASP data for mixture POSCARS in separate directories for each compound type
def loaddata(directory):   ## Need to edit based on where you put the data 
    filenames = []
    structnames = []
    for subdir in glob(directory+"/*/", recursive = True):
       for filename in glob(subdir+"*"):
          fnsplit = filename.split("/")
          structname = fnsplit[6].split("_")[0]+fnsplit[7].split("_")[1]+fnsplit[7].split("_")[2]
          filenames.append(filename)
          structnames.append(structname)
    return filenames, structnames

#directory = "/Users/alukas/Desktop/Materials-Research-Lab/data/mixtures"
#filenames, structnames = loaddata(directory)
#print(filenames)
#print(structnames)  
### ---- Calculate structure fingerprints.

def calc_cnn(struct,idx):
   atomic_masses = {"O":16,"C":12,"F":18,"Li":3}
   chem_dict = {"atomic masses":atomic_masses}
   op_types = load_ops()
   #cnfp = CrystalNNFingerprint(op_types, distance_cutoffs=None, x_diff_weight=0,chem_info=chem_dict)
   cnfp = CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0)
   return cnfp.featurize(struct, idx)

def calc_ssfp(struct, chem=False):
   atomic_masses = {"O":16,"C":12,"F":18,"Li":3}
   chem_dict = {"atomic masses":atomic_masses}
   op_types = load_ops()
   if chem:
       cnfp = CrystalNNFingerprint(op_types, distance_cutoffs=None, x_diff_weight=0,chem_info=chem_dict)
   else:
       cnfp = CrystalNNFingerprint(op_types, distance_cutoffs=None, x_diff_weight=0)
   ssfp = PartialsSiteStatsFingerprint(
      #cnfp,
      cnfp, stats=('mean'),
      #cnfp, stats=('mean', 'std_dev', 'minimum', 'maximum'))
      #CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0,chem_info=chem_dict),
      #CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
      include_elems=("Li","O","C","F"),exclude_elems=())
      #stats=('mean', 'std_dev', 'minimum', 'maximum'),include_elems=("Li","F","O"),exclude_elems=("C"))
   ssfp.fit([struct]) #Add in chemistry
   #cnfp = CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0)
   #cnfp.featurize(struct, 0)
   v_struct = np.array(ssfp.featurize(struct)) #Here is where we fit "struct" to all shapes loaded in op_types
   return v_struct

def calc_ssf(struct):
    atomic_masses = {"O":16,"C":12,"F":18,"Li":3}
    chem_dict = {"atomic masses":atomic_masses}
    op_types = load_ops()
    cnfp = CrystalNNFingerprint(op_types, distance_cutoffs=None, x_diff_weight=0,chem_info=chem_dict)
    #cnfp = CrystalNNFingerprint(op_types, distance_cutoffs=None, x_diff_weight=0)
    ssf = SiteStatsFingerprint(
        #cnfp)
        cnfp, stats=('mean', 'std_dev', 'minimum', 'maximum'))
        #CrystalNNFingerprint.from_preset('ops',distance_cutoffs=None, x_diff_weight=0),
    v_struct = np.array(ssf.featurize(struct))
    return v_struct

def load_ops():
    """
    Load the file for the op types

    Returns:
        (dict)
    """
    with open("/Users/alukas/miniconda3/envs/matminer_env/lib/python3.9/site-packages/matminer/featurizers/site/cn_target_motif_op.yaml") as f:
        cn_target_motif_op =  yaml.YAML(typ="safe", pure=True).load(f)
    op_types = copy.deepcopy(cn_target_motif_op)
    for k in range(24):
       if k + 1 in op_types:
          op_types[k + 1].insert(0, "wt")
       else:
          op_types[k + 1] = ["wt"]
    return op_types

def featureize_site(structure,idx):
    return idx

    #print('Distance between struct 1 and 2: {:.4f}'.format(np.linalg.norm(v_struct1 - v_struct2)))

if __name__ == "__main__":
    main()