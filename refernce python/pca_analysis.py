import os
import json
from glob import glob
import numpy as np
from mp_api.client import MPRester
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
import pymatgen.core
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.local_env import site_is_of_motif_type
from sklearn.decomposition import PCA
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn import datasets
#Try after removing 1362
def main():
# License: BSD 3 clause
    #X = np.load("fingerprints/fingerprints_all_chem_stats.npy")
    X = np.load("fingerprints/fingerprints_all_chem_stats.v1.npy")
    #X0 = np.load("pfingerprints_all.npy")
    sites = np.load("fingerprints/fingerprints_int.npy")
    #int0 = np.load("fing_int101001.npy")
    #int1 = np.load("fing_int101001.npy")
    #X = np.vstack((X0,int0,int1))
    #y = np.asarray(545*[5]+403*[4]+499*[3]+2*[1])
    #y = np.asarray(545*[5]+403*[4]+499*[3])
    #y = np.asarray(545*[5]+403*[4]+498*[3]) # Label Number of structs w/ Li5, Li4, Li3 
    #y = np.asarray(545*[5]+403*[4]+499*[3])
    #y = np.asarray(144*[5]+34*[4]+70*[3])
    n_neighbors = 3 #Nicole forgets what this does.
    random_state = 0
    #X = np.vstack((X,sites))
    X = np.vstack((sites,X))
    print(X.shape)
    y = np.asarray(547*[5]+403*[4]+498*[3]) # Label Number of structs w/ Li5, Li4, Li3 
    # Load Digits dataset
    #X, y = datasets.load_digits(return_X_y=True)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=random_state
    )

    dim = len(X[0])
    n_classes = len(np.unique(y))

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))

    # Reduce dimension to 2 with LinearDiscriminantAnalysis
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
    )

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Make a list of the methods to be compared
    dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()
        # plt.subplot(1, 3, i + 1, aspect=1)

        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)
        y_vals = model.transform(sites)
        y_pred = knn.predict(y_vals)
        print("knn",y_pred)

        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)
        findNN = 6
        if i==0:
            distint = np.linalg.norm(X_embedded - y_vals[1],axis=1)
            print(np.argpartition(distint,4)[:findNN]) #0 [ 207 1088  536  140  541  205]
            idx = np.argpartition(distint,4) #1 [ 207  205 1088  541  536  140]
            print(distint[idx[:findNN]]) #no outlier 1 [1088  207  541  205  341  536]
            np.save("PCA_values.npy",X_embedded)

        # Plot the projected points and show the evaluation score
        plt.plot(y_vals[0,0],y_vals[0,1],marker='+',c="b",markersize=10)
        plt.plot(y_vals[1,0],y_vals[1,1],marker='*',c="c",markersize=10)
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
        plt.legend(*scatter.legend_elements())
        plt.title(
            "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
        )
    plt.show() 
    #filenames,structnames = loaddata()
       #print(structname)
       #structure = Structure.from_file(filename)
       #threshm = {"qtet":0.3, "qoct":0.5,"qbcc":0.5,"q6":0.4,"qsqpyr":0.8,"qtribipyr":0.8}
       #motifdict = calc_motif(structure,threshm)
       #print(motifdict)
       #fingprints.append(calc_ssf(structure))
    #np.save("fingerprints.out",np.asarray(fingprints))
#    with open("crystalNNfing.json", "w") as outfile: 
#        json.dump(fingdict, outfile)
    #fingarray = calc_ssf(structure)
    #print(fingarray.shape)
#filenames = sorted(glob('unitcells/LiF.POSCAR*'))
#structure1 = Structure.from_file("/g/g22/adelsten/SCEI/mixtures/Li3CO3F_crystal_structures/POSCAR_z4_-6.09517")
#structure2 = Structure.from_file("/g/g22/adelsten/SCEI/mixtures/Li3CO3F_crystal_structures/POSCAR_z2_-6.09486")
#print(structure1,structure2)

### ---- Load in VASP data
def loaddata():
    filenames = []
    structnames = []
    directory = "/g/g22/adelsten/SCEI/mixtures"
    for subdir in glob(directory+"/*/", recursive = True):
       for filename in glob(subdir+"*"):
          fnsplit = filename.split("/")
          structname = fnsplit[6].split("_")[0]+fnsplit[7].split("_")[1]+fnsplit[7].split("_")[2]
          filenames.append(filename)
          structnames.append(structname)
    return filenames, structnames

### ---- Calculate motifs
def calc_motif(struct,threshm):
    allunrec = True
    motifdict = {}
    for i in range(len(struct.sites)):
        motif = site_is_of_motif_type(struct,i,approach="min_VIRE",delta=0.2,thresh=threshm)
        elem = struct.sites[i].species_string
        if motif!="unrecognized":
            motifdict[i] = [elem,motif] 
            #print("{} {} {}".format(i+1, elem, motif))
            allunrec = False
    if allunrec:
        motifdict[i] = [motif]
    return motifdict
      

### ---- Calculate structure fingerprints.
def calc_ssf(struct):
   ssf = SiteStatsFingerprint(
      CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
      stats=('mean', 'std_dev', 'minimum', 'maximum'))
   v_struct = np.array(ssf.featurize(struct))
   return v_struct

#print('Distance between struct 1 and 2: {:.4f}'.format(np.linalg.norm(v_struct1 - v_struct2)))

### ---- Calculate 

#with MPRester("anbiBn745WPl8caNbbj66bt1oKkddNUI") as mpr:

    # Get structures.
#    diamond = mpr.get_structure_by_material_id("mp-66")
#    gaas = mpr.get_structure_by_material_id("mp-2534")
#    print(diamond, gaas)
    # Calculate structure fingerprints.
    #ssf = SiteStatsFingerprint(
    #    CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
    #    stats=('mean', 'std_dev', 'minimum', 'maximum'))
    #v_diamond = np.array(ssf.featurize(diamond))
    #v_gaas = np.array(ssf.featurize(gaas))

    # Print out distance between structures.
    #print('Distance between diamond and GaAs: {:.4f}'.format(np.linalg.norm(v_diamond - v_gaas)))

if __name__ == "__main__":
    main()
