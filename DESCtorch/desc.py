### yu xiaokang 2022-04-01 pytorch 

from __future__ import division
from __future__ import print_function
import os,math
os.environ['PYTHONHASHSEED'] = '0'
import matplotlib
havedisplay = "DISPLAY" in os.environ
#if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
from time import time as get_time
import random
import numpy as np
import pandas as pd
#import tensorflow as tf
import multiprocessing
from anndata import AnnData
import scanpy as sc
from scipy.sparse import issparse

from .network import *
#or 
import torch
import random
from torch.utils.data import DataLoader

######################## set device and random seed ###########################
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.badatahmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
###############################################################################



def getdims(x=(10000,200)):
    """
    This function will give the suggested nodes for each encoder layer
    return the dims for network
    """
    assert len(x)==2
    n_sample=x[0]
    if n_sample>20000:# may be need complex network
        dims=[x[-1],128,32]
    elif n_sample>10000:#10000
        dims=[x[-1],64,32]
    elif n_sample>5000: #5000
        dims=[x[-1],32,16] #16
    elif n_sample>2000:
        dims=[x[-1],128]
    elif n_sample>500:
        dims=[x[-1],64]
    else:
        dims=[x[-1],16]
    return dims


def train_single(data,dims=None,
        alpha=1.0,
        tol=0.005,
        init='xavier_uniform',
        louvain_resolution=1.0,
        n_neighbors=15,
        pretrain_epochs=300,
        batch_size=256,
        activation=nn.ReLU(),
        actincenter=nn.Tanh(),
        drop_rate_SAE=0.2,
        is_stacked=True,
        use_earlyStop=True,
        use_ae_weights=False,
	save_encoder_weights=False,
        save_encoder_step=4,
        save_dir='result_tmp',
        max_iter=1000,
        epochs_fit=10, 
        device=torch.device("cpu"),
        use_GPU=True,
        GPU_id=None,
        random_seed=201809,
        verbose=True,
	do_tsne=False,
	learning_rate=150,
	perplexity=30,
        do_umap=False,
        kernel_clustering="t"
):
    #print("running train_single fucntion ...")
    if isinstance(data,AnnData):
        adata=data
    else:
        adata=sc.AnnData(data)
    #make sure dims 
    if dims is None:
        dims=getdims(adata.shape)
    #print("dims[0]={},adata.shape[-1]={}".format(dims[0],adata.shape[-1])) ,当resolution为多个时，会出问题
    assert dims[0]==adata.shape[-1],'the number of columns of x doesnot equal to the first element of dims, we must make sure that dims[0]==x.shape[0]'
    
    seed_torch(random_seed)
    total_cpu=multiprocessing.cpu_count()
    print('The number of cpu in your computer is',total_cpu)

    if use_GPU and torch.cuda.is_available():
        #if you use GPU,you must be sure that there is GPU in your device
        if(GPU_id is not None):
            device=torch.device("cuda"+str(GPU_id))
        else:
            device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    
    #################################################################
    if not use_ae_weights and os.path.isfile(os.path.join(save_dir,"autoencoder_model.pkl")):
        os.remove(os.path.join(save_dir,"autoencoder_model.pkl"))
  
    tic=get_time()#recored time         
    desc=DescModel(dims=dims,
              x=adata.X,
              alpha=alpha,
              tol=tol,
              init=init,
              louvain_resolution=louvain_resolution,
              n_neighbors=n_neighbors,
              pretrain_epochs=pretrain_epochs,
              epochs_fit=epochs_fit,
              batch_size=batch_size,
              activation=activation,
              actincenter=actincenter,
              drop_rate_SAE=drop_rate_SAE,
              is_stacked=is_stacked,
              use_earlyStop=use_earlyStop,
              use_ae_weights=use_ae_weights,
	      save_encoder_weights=save_encoder_weights,
              save_encoder_step=save_encoder_step,
              save_dir=save_dir,
              kernel_clustering=kernel_clustering,
              device=device,
              verbose=verbose
    )
    ds_train = torch.utils.data.TensorDataset(torch.FloatTensor(adata.X))
    Embeded_z=desc.get_embedding(ds_train,desc.dec_model)
    q_pred=desc.predict(ds_train,desc.dec_model,return_prob=True).data.numpy()
    print("The desc has been trained successfully!!!!!!")

    print("The runtime of (resolution="+str(louvain_resolution)+")is:",get_time()-tic)
    y_pred=pd.Series(np.argmax(q_pred,axis=1),index=adata.obs.index,dtype='category')
    y_pred.cat.categories=list(range(len(y_pred.unique())))
    adata.obs['desc_'+str(louvain_resolution)]=y_pred
    adata.obsm['X_Embeded_z'+str(louvain_resolution)]=Embeded_z
    if do_tsne:
        sc.tl.tsne(adata,use_rep="X_Embeded_z"+str(louvain_resolution),learning_rate=learning_rate,perplexity=perplexity)
        adata.obsm["X_tsne"+str(louvain_resolution)]=adata.obsm["X_tsne"].copy()
        print('tsne finished and added X_tsne'+str(louvain_resolution),' into the umap coordinates (adata.obsm)\n')
        #sc.logging.msg(' tsne finished', t=True, end=' ', v=4)
        #sc.logging.msg('and added\n'
        #         '    \'X_tsne\''+str(louvain_resolution),'the tsne coordinates (adata.obs)\n', v=4)
    if do_umap:
        sc.pp.neighbors(adata,n_neighbors=n_neighbors,use_rep="X_Embeded_z"+str(louvain_resolution))
        sc.tl.umap(adata)
        adata.obsm["X_umap"+str(louvain_resolution)]=adata.obsm["X_umap"].copy()
        print('umap finished and added X_umap'+str(louvain_resolution),' into the umap coordinates (adata.obsm)\n')
        #sc.logging.msg(' umap finished', t=True, end=' ', v=4)
        #sc.logging.msg('and added\n'
        #        '    \'X_umap\''+str(louvain_resolution),'the umap coordinates (adata.obsm)\n', v=4)
        del adata.uns["neighbors"]

    #prob_matrix
    adata.uns['prob_matrix'+str(louvain_resolution)]=q_pred
    return adata


def train(data,dims=None,
        alpha=1.0,
        tol=0.005,
        init='xavier_uniform',
        louvain_resolution=[0.6,0.8],
        n_neighbors=10,
        pretrain_epochs=300,
        batch_size=256,
        activation=nn.ReLU(),
        actincenter=nn.Tanh(),
        drop_rate_SAE=0.2,
        is_stacked=True,
        use_earlyStop=True,
        use_ae_weights=True,
	save_encoder_weights=False,
        save_encoder_step=5,
        save_dir='result_tmp',
        max_iter=1000,
        epochs_fit=5,
        device=torch.device("cpu"),
        random_seed=201809,
        use_GPU=False,
        GPU_id=None,
        verbose=True,
	do_tsne=False,
	learning_rate=150,
	perplexity=30,
        do_umap=False,
        kernel_clustering="t"
): 
    """ Deep Embeded single cell clustering(DESC) API
    Conduct clustering for single cell data given in the anndata object or np.ndarray,sp.sparmatrix,or pandas.DataFrame
      
    
    Argument:
    ------------------------------------------------------------------
    data: :class:`~anndata.AnnData`, `np.ndarray`, `sp.spmatrix`,`pandas.DataFrame`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    dims: `list`, the number of node in encoder layer, which include input dim, that is
    [1000,64,32] represents for the input dimension is 1000, the first hidden layer have 64 node, and second hidden layer(or bottle neck layers) have 16 nodes. if not specified, it will be decided automatically according to the sample size.
    
    alpha: `float`, optional. Default: `1.0`, the degree of t-distribution.
    tol: `float`, optional. Default: `0.005`, Stop criterion, clustering procedure will be stoped when the difference ratio betwen the current iteration and last iteration larger than tol.
    init: `str`,optional. Default: `glorot_uniform`.
        Initialization method used to initialize weights.

    louvain_resolution: `list  or str or float. for example, louvain_resolution=1.2 or louvain_resolution=[0.2,0.4,0.8] or louvain_resolution="0.3,0.4,0.8" sep with ","
    n_neighbors, `int`, optional. Default: 10. The size of local neighborhood (in terms of number of neighboring data points) used for connectivity matrix. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. In general values should be in the range 2 to 100. Lo 

    pretrain_epochs:'int',optional. Default:`300`,the number of epochs for autoencoder model. 

    batch_size: `int`, optional. Default:`256`, the batch size for autoencoder model and clustering model. 

    activation; `str`, optional. Default,`relu`. the activation function for autoencoder model,which can be 'elu,selu,softplus,tanh,siogmid et al.', for detail please refer to`https://keras.io/activations/`.

    actincenter: `str`, optional. Default,'tanh', the activation function for the last layer in encoder and decoder model.

    drop_rate_SAE: `float`, optional. Default, `0.2`. The drop rate for Stacked autoencoder, which just for  finetuning. 

    is_stacked:`bool`,optional. Default,`True`.The model wiil be pretrained by stacked autoencoder if is_stacked==True.

    use_earlyStop:`bool`,optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.

    use_ae_weights: `bool`, optional. Default, `True`. Whether use ae_weights that has been pretrained(which must saved in `save_dir/ae_weights.h5`)

    save_encoder_weights: `bool`, optional. Default, `False`, it will save inter_ae_weights for every 20 iterations. )

    save_dir: 'str',optional. Default,'result_tmp',some result will be saved in this directory.

    max_iter: `int`, optional. Default,`1000`. The maximum iteration for clustering.

    epochs_fit: `int or fload`,optional. Default,`4`, updateing clustering probability for each epochs_fit*n_sample, where n_sample is the sample size 

    verbose,`bool`, optional. Default, `True`. It will ouput the model summary if verbose==True.

    do_tsne,`bool`,optional. Default, `False`. Whethter do tsne for representation or not.

    learning_rate,`float`,optional, Default(150).Note that the R-package "Rtsne" uses a default of 200. The learning rate can be a critical parameter. It should be between 100 and 1000. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high. If the cost function gets stuck in a bad local minimum increasing the learning rate helps sometimes.

    perplexity, `float`, optional, Default(30). The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE is quite insensitive to this parameter.
    do_umap, `bool`, optional. Default, `False`,Whethter do umap for representation or not
    ------------------------------------------------------------------
    """
    print("===============runnning desc.train function...............")
    if isinstance(data,AnnData):
        adata=data
    elif isinstance(data,pd.DataFrame):
        adata=sc.AnnData(data,obs=data.index,var=data.columns)
    else:
        x=data.toarray() if issparse(data) else data
        adata=sc.AnnData(x)

    if dims is None:
        dims=getdims(adata.shape)

    if isinstance(louvain_resolution,float) or isinstance(louvain_resolution,int):
        louvain_resolution=[float(louvain_resolution)]
    elif  isinstance(louvain_resolution,str):
        louvain_resolution=list(map(float,louvain_resolution.split(",")))
    else:
        assert isinstance(louvain_resolution,list),'louvain_resolution must be either a string with spearator "," or a list like [1.0,2.0,3.0] '
        louvain_resolution=list(map(float,louvain_resolution))
    time_start=get_time()
    for ith,resolution in enumerate(louvain_resolution):
        print("Start to process resolution=",str(resolution))
        use_ae_weights=use_ae_weights if ith==0 else True
        res=train_single(data=data,
            dims=dims,
            alpha=alpha,
            tol=tol,
            init=init,
            louvain_resolution=resolution,
            n_neighbors=n_neighbors,
            pretrain_epochs=pretrain_epochs,
            epochs_fit=epochs_fit,
            batch_size=batch_size,
            activation=activation,
            actincenter=actincenter,
            drop_rate_SAE=drop_rate_SAE,
            is_stacked=is_stacked,
            use_earlyStop=use_earlyStop,
            use_ae_weights=use_ae_weights,
	    save_encoder_weights=save_encoder_weights,
            save_encoder_step=save_encoder_step,
            save_dir=save_dir,
            max_iter=max_iter,
            device=device,
            use_GPU=use_GPU,
            GPU_id=GPU_id,
            verbose=verbose,
	    do_tsne=do_tsne,
	    learning_rate=learning_rate,
	    perplexity=perplexity,
            do_umap=do_umap,
            kernel_clustering=kernel_clustering)
        #update adata
        data=res
    print("The run time for all resolution is:",get_time()-time_start)
    print("After training, the information of adata is:\n",data)
    return data

