import os
from desc import *
import pandas as pd
import numpy as np
import scanpy as sc
from time import time
import sys
import matplotlib
import matplotlib.pyplot as plt
import random
sc.settings.set_figure_params(dpi=300)

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader

###########clean screen #######
# import os
# os.system('clear')
import matplotlib 
#%matplotlib inline
#matplotlib.use('Agg')# 
#####################


####################### read data and data preprocessing#############
adata=sc.datasets.paul15()
adata.obs['celltype']=adata.obs['paul15_clusters'].str.split("[0-9]{1,2}", n = 1, expand = True).values[:,1]
adata.obs['celltype2']=adata.obs['paul15_clusters']
sc.pp.log1p(adata)
#sc.pp.filter_genes_dispersion(adata,n_top_genes=1000) #older scanpy
sc.pp.highly_variable_genes(adata,n_top_genes=1000,subset=True,inplace=True)
sc.pp.scale(adata,max_value=6)# if the the dataset has two or more batches you can use `adata=desc.scale(adata,groupby="BatchID")`



#################### desc training #######################
dataset="paul15"
save_dir="./result/"+dataset+"_result"
reso=0.8
sc.settings.figdir=save_dir
filelist = [ f for f in os.listdir(os.path.join(os.getcwd(),save_dir))]
for f in filelist:
    os.remove(os.path.join(os.getcwd(),save_dir, f))
#############################################################################

#################### desc training #######################
sc.settings.figdir=save_dir
adata=desc.train(adata,
        dims=[adata.shape[1],64,32],
        tol=0.005,
        n_neighbors=10,
        batch_size=256,
        louvain_resolution=[reso],# not necessarily a list, you can only set one value, like, louvain_resolution=1.0
        save_dir=str(save_dir),
        do_tsne=True,
        learning_rate=200, # the parameter of tsne
        device=torch.device("cpu"),
        save_encoder_weights=False,
        save_encoder_step=3,# save_encoder_weights is False, this parameter is not used
        use_ae_weights=False,
        do_umap=False) #if do_uamp is False, it will don't compute umap coordiate
####################################################################################


############################## Visulizing the result of desc #######################
adata.obs['max.prob']=adata.uns["prob_matrix"+str(reso)].max(1)
sc.pl.scatter(adata,basis="tsne"+str(reso),color=['desc_'+str(reso),'max.prob'],save="_tsne"+str(reso)+"_"+dataset+"_prob.png")
sc.pl.scatter(adata,basis="tsne"+str(reso),color=['desc_'+str(reso),'celltype','celltype2'],save="_tsne"+str(reso)+"_"+dataset+"_desc.png")
print("done")
####################################################################################
