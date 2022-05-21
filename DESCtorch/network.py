"""
pytorch implement Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis
"""
from __future__ import division
import os
import matplotlib
havedisplay = "DISPLAY" in os.environ
#if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from time import time as get_time
import numpy as np
import random
from sklearn.cluster import KMeans
#import tensorflow as tf
import scanpy as sc
import pandas as pd
from natsort import natsorted #call natsorted
import os
from .pytorchtools import EarlyStopping
from .sdae import StackedDenoisingAutoEncoder  # this is for installing package
from .dae import DenoisingAutoencoder
import torch.nn as nn
from tqdm import tqdm
import torch
from typing import Any, Callable, Optional
import math
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim import SGD
import torch.nn.functional as F
from .dec import DEC,target_distribution

class DescModel(object):
    def __init__(self,
                 dims,
                 x, # input matrix, row sample, col predictors 
                 alpha=1.0,
		         tol=0.005,
                 init='xavier_uniform', #initialization method
                 louvain_resolution=1.0, # resolution for louvain 
                 n_neighbors=10,    # the 
                 pretrain_epochs=300, # epoch for autoencoder
                 epochs_fit=4, #epochs for each update,int or float 
                 batch_size=256, #batch_size for autoencoder
		         activation=nn.ReLU(),
                 actincenter=nn.Tanh(),# activation for the last layer in encoder, and first layer in the decoder 
                 drop_rate_SAE=0.2,
                 is_stacked=True,
                 use_earlyStop=True,
                 use_ae_weights=False,
		         save_encoder_weights=False,
                 save_encoder_step=5,
                 save_dir="result_tmp",
                 kernel_clustering="t",
                 device=torch.device("cpu"),
                 verbose=True
                 # save result to save_dir, the default is "result_tmp". if recurvie path, the root dir must be exists, or there will be something wrong: for example : "/result_singlecell/dataset1" will return wrong if "result_singlecell" not exist
                 ):

        if not os.path.exists(save_dir):
            print("Create the directory:"+str(save_dir)+" to save result")
            os.mkdir(save_dir)
        self.dims = dims
        self.x=x #feature n*p, n:number of cells, p: number of genes
        self.alpha = alpha
        self.tol=tol
        self.init=init
        self.input_dim = dims[0]  # for clustering layer 
        self.n_stacks = len(self.dims) - 1
        self.is_stacked=is_stacked
        self.resolution=louvain_resolution
        self.n_neighbors=n_neighbors
        self.pretrain_epochs=pretrain_epochs
        self.epochs_fit=epochs_fit
        self.batch_size=batch_size
        self.activation=activation
        self.actincenter=actincenter
        self.drop_rate_SAE=drop_rate_SAE
        self.is_stacked=is_stacked
        self.use_earlyStop=use_earlyStop
        self.use_ae_weights=use_ae_weights
        self.save_encoder_weights=save_encoder_weights
        self.save_encoder_step=save_encoder_step
        self.save_dir=save_dir
        self.kernel_clustering=kernel_clustering

        self.autoencoder=StackedDenoisingAutoEncoder(
        self.dims, activation=activation,final_activation=actincenter,gain=1.0,
    ) 
        ################## print model ###########
        if(verbose):
            print("constructing DESC model")
            print(self.autoencoder)
        ##########################################
        
        ds_train = torch.utils.data.TensorDataset(torch.FloatTensor(self.x))
        t0 = get_time()
        self.pretrain(ds_train,epochs=self.pretrain_epochs/2,batch_size=self.batch_size,corruption=self.drop_rate_SAE,device=device,verbose=verbose)
        self.finetune_autoencoder(ds_train,epochs=self.pretrain_epochs,batch_size=self.batch_size,corruption=None,device=device,verbose=verbose) 
        print('Pretraining time is', get_time() - t0)
        #save ae results into disk
        #save autoencoder model
        torch.save(self.autoencoder,os.path.join(self.save_dir,"autoencoder_model.pkl"))

        dataloader = DataLoader(
            ds_train, batch_size=self.batch_size, pin_memory=False, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
        features = []
        if isinstance(self.autoencoder.encoder, torch.nn.Module):
            self.autoencoder.encoder.eval()
        for batch in data_iterator:
            if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
                batch = batch[0]
            batch = batch.squeeze(1).view(batch.size(0), -1)
            batch = batch.to(device)
            output = self.autoencoder.encoder(batch)
            features.append(
                output.detach().cpu()
            )  # move to the CPU to prevent out of memory on the GPU
        features=torch.cat(features).cpu().numpy()

        print("...number of clusters is unknown, Initialize cluster centroid using louvain method")
        # can be replaced by other clustering methods
        # using louvain methods in scanpy
        adata0 = sc.AnnData(features)
        if adata0.shape[0] > 200000:
            np.random.seed(adata0.shape[0])  # set  seed
            adata0 = adata0[np.random.choice(adata0.shape[0], 200000, replace=False)]
        sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors, use_rep="X")
        sc.tl.louvain(adata0, resolution=self.resolution)
        sc.tl.umap(adata0)
        if(verbose):
            sc.pl.umap(adata0, color=["louvain"], save="_init_louvain_" + str(self.resolution) + ".png")

        Y_pred_init = adata0.obs['louvain']
        init_pred = np.asarray(Y_pred_init, dtype=int)
        if np.unique(init_pred).shape[0]<=1:
        # avoid only a cluster
        # print(np.unique(self.init_pred))
            exit("Error: There is only a cluster detected. The resolution:"+str(self.resolution)+"is too small, please choose a larger resolution!!")
        features = pd.DataFrame(adata0.X, index=np.arange(0, adata0.shape[0]))
        Group = pd.Series(init_pred, index=np.arange(0, adata0.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        n_clusters = cluster_centers.shape[0]
        init_centroid = [cluster_centers]
        predicted = init_pred.copy()

        self.dec_model = DEC(cluster_number=n_clusters, hidden_dimension=self.dims[-1], encoder=self.autoencoder.encoder)
        self.dec_train(dataset=ds_train,init_pred=predicted,cluster_centers=cluster_centers,epochs=1000,epochs_fit=self.epochs_fit,batch_size=self.batch_size,stopping_delta=self.tol,device=device,verbose=verbose)

    def pretrain(self,
            dataset,
            epochs: int,
            batch_size: int,
            corruption: Optional[float] = 0.2,
            device=torch.device("cpu"),
            verbose=True
    ) -> None:
        """
        Given an autoencoder, train it using the data provided in the dataset; for simplicity the accuracy is reported only
        on the training dataset. If the training dataset is a 2-tuple or list of (feature, prediction), then the prediction
        is stripped away.

        :param dataset: instance of Dataset to use for training
        :param autoencoder: instance of an autoencoder to train
        :param epochs: number of training epochs
        :param batch_size: batch size for training
        :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
        :return: None
        """
        current_dataset = dataset
        number_of_subautoencoders = len(self.autoencoder.dimensions) - 1
        for index in range(number_of_subautoencoders):
            encoder, decoder = self.autoencoder.get_stack(index)
            embedding_dimension = self.autoencoder.dimensions[index]
            hidden_dimension = self.autoencoder.dimensions[index + 1]
            # manual override to prevent corruption for the last subautoencoder
            # if index == (number_of_subautoencoders - 1):
            #     corruption = None
            # initialise the subautoencoder
            sub_autoencoder = DenoisingAutoencoder(
                embedding_dimension=embedding_dimension,
                hidden_dimension=hidden_dimension,
                activation=self.activation,
                actincenter=self.actincenter,
                corruption=nn.Dropout(corruption),
                gain=1.0
            )
            if(verbose):
                print(sub_autoencoder)
            print("train {}th sub autoencoder".format(index+1))
            self.train_sub_autoencoder( 
                current_dataset,
                sub_autoencoder,
                epochs,
                batch_size,
                corruption=None,# already have dropout in the DAE,have dropout in DenosingAutoencoder,
                device=device,
                verbose=verbose
            )
            # copy the weights
            sub_autoencoder.copy_weights(encoder, decoder)
            # pass the dataset through the encoder part of the subautoencoder
            if index != (number_of_subautoencoders - 1):
                current_dataset = TensorDataset(
                    self.exact_features(
                        current_dataset,
                        sub_autoencoder,
                        batch_size,
                        device=device
                    )
                )

    def train_sub_autoencoder(self,
            dataset: torch.utils.data.Dataset,
            autoencoder: torch.nn.Module,
            epochs: int,
            batch_size: int,
            validation: Optional[torch.utils.data.Dataset] = None,
            corruption: Optional[float] = None,
            device=torch.device('cpu'),
            verbose=True,
    ) -> None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=True ,
            num_workers=0,
        )
        validation_loader = None
        loss_function = nn.MSELoss()
        autoencoder=autoencoder.to(device)
        autoencoder.train()
        validation_loss_value = -1
        loss_value = 0

        decaying_step = 3
        for j in range(int(decaying_step)):  # learning rate multiplies 0.1 every 'epochs/decaying_step' epochs
            print('learning rate =', pow(10, -1 - j))
            early_stopping = EarlyStopping(patience=10, delta=1e-4, verbose=True,path=self.save_dir+"/"+"ae_earlystopping.pkl")
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=pow(10, -1 - j),
                                  momentum=0.9)
            train_loss = []
            for epoch in range(math.ceil(epochs / decaying_step)):
                train_temp_loss = []
                data_iterator = tqdm(
                    dataloader,
                    leave=True,
                    unit="batch",
                    postfix={"epo": epoch, "train_loss": "%.6f" % 0.0},
                )
                for index, batch in enumerate(data_iterator):
                    if (
                            isinstance(batch, tuple)
                            or isinstance(batch, list)
                            and len(batch) in [1, 2]
                    ):
                        batch = batch[0].to(device)
                    # run the batch through the autoencoder and obtain the output
                    if corruption is not None:
                        output = autoencoder(F.dropout(batch, corruption))
                    else:
                        output = autoencoder(batch)
                    loss = loss_function(output, batch)
                    # accuracy = pretrain_accuracy(output, batch)
                    loss_value = float(loss.item())
                    train_temp_loss.append(loss_value)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step(closure=None)
                    data_iterator.set_postfix(
                        epo=epoch, train_loss="%.6f" % loss_value
                    )
                # train_loss.append(train_temp_loss)
                train_loss.append(np.mean(train_temp_loss))
                early_stopping(train_loss[-1], autoencoder)  #
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    def finetune_autoencoder(
            self,
            dataset: torch.utils.data.Dataset,
            epochs: int,
            batch_size: int,
            optimizer: Optional[torch.optim.Optimizer]=None,
            scheduler: Any = None,
            validation: Optional[torch.utils.data.Dataset] = None,
            corruption: Optional[float] = None,
            cuda: bool = False,
            sampler: Optional[torch.utils.data.sampler.Sampler] = None,
            silent: bool = False,
            update_freq: Optional[int] = 1,
            update_callback: Optional[Callable[[float, float], None]] = None,
            num_workers: Optional[int] = None,
            epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
            device=torch.device("cpu"),
            verbose=True,
    ) -> None:
        """
        Function to train an autoencoder using the provided dataset. If the dataset consists of 2-tuples or lists of
        (feature, prediction), then the prediction is stripped away.

        :param dataset: training Dataset, consisting of tensors shape [batch_size, features]
        :param autoencoder: autoencoder to train
        :param epochs: number of training epochs
        :param batch_size: batch size for training
        :param optimizer: optimizer to use
        :param scheduler: scheduler to use, or None to disable, defaults to None
        :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
        :param validation: instance of Dataset to use for validation, set to None to disable, defaults to None
        :param cuda: whether CUDA is used, defaults to True
        :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
        :param silent: set to True to prevent printing out summary statistics, defaults to False
        :param update_freq: frequency of batches with which to update counter, set to None disables, default 1
        :param update_callback: optional function of loss and validation loss to update
        :param num_workers: optional number of workers to use for data loading
        :param epoch_callback: optional function of epoch and model
        :return: None
        """
        print("finetune autoencoder ....")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            sampler=sampler,
            shuffle=True if sampler is None else False,
            num_workers=num_workers if num_workers is not None else 0,
        )

        validation_loader = None

        loss_function = nn.MSELoss()
        self.autoencoder=self.autoencoder.to(device)
        self.autoencoder.train()
        validation_loss_value = -1
        loss_value = 0

        print('Fine-tuning autoencoder end-to-end')
        for j in range(math.ceil(epochs / 50)):
            lr = pow(10, -j)
            print('learning rate =', lr)
            early_stopping = EarlyStopping(patience=10, delta=1e-4, verbose=True,path=self.save_dir+"/"+"ae_earlystopping.pkl")
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.autoencoder.parameters()), lr=pow(10, -j),
                                  momentum=0.9)
            train_loss = []
            for epoch in range(50):
                train_temp_loss = []
                data_iterator = tqdm(
                    dataloader,
                    leave=True,
                    unit="batch",
                    postfix={"epo": epoch, "train_loss": "%.6f" % 0.0},
                    disable=silent,
                )
                for index, batch in enumerate(data_iterator):
                    if (
                            isinstance(batch, tuple)
                            or isinstance(batch, list)
                            and len(batch) in [1, 2]
                    ):
                        batch = batch[0].to(device)
                    if cuda:
                        batch = batch.cuda(non_blocking=True)
                    # run the batch through the autoencoder and obtain the output
                    if corruption is not None:
                        output = self.autoencoder(F.dropout(batch, corruption))
                    else:
                        output = self.autoencoder(batch)
                    loss = loss_function(output, batch)
                    loss_value = float(loss.item())
                    train_temp_loss.append(loss_value)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step(closure=None)
                    data_iterator.set_postfix(
                        epo=epoch, train_loss="%.6f" % loss_value
                    )

                train_loss.append(np.mean(train_temp_loss))
                early_stopping(train_loss[-1],self.autoencoder)  #
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    def dec_train(self,
            init_pred:None,
            cluster_centers:None,
            dataset: torch.utils.data.Dataset,
            epochs: int,
            epochs_fit:5,
            batch_size:256,
            stopping_delta: Optional[float] = 0.005,
            device=torch.device("cpu"),
            verbose=True,
    ) -> None:
        """
        Train the DEC model given a dataset, a model instance and various configuration parameters.

        :param dataset: instance of Dataset to use for training
        :param model: instance of DEC model to train
        :param epochs: number of training epochs
        :param batch_size: size of the batch to train with
        :param optimizer: instance of optimizer to use
        :param stopping_delta: label delta as a proportion to use for stopping, None to disable, default None
        :param collate_fn: function to merge a list of samples into mini-batch
        :param cuda: whether to use CUDA, defaults to True
        :param sampler: optional sampler to use in the DataLoader, defaults to None
        :param silent: set to True to prevent printing out summary statistics, defaults to False
        :param update_freq: frequency of batches with which to update counter, None disables, default 10
        :param evaluate_batch_size: batch size for evaluation stage, default 1024
        :param update_callback: optional function of accuracy and loss to update, default None
        :param epoch_callback: optional function of epoch and model, default None
        :return: None
        """
        print("dec training......")
        predicted=init_pred.copy()
        self.dec_model=self.dec_model.to(device)
        self.dec_model.train()
        y_pred_last=init_pred.copy()
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)

        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True)

        with torch.no_grad():
            # initialise the cluster centers
            self.dec_model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
        
        loss_function = torch.nn.KLDivLoss(size_average=False)
        optimizer = SGD(self.dec_model.parameters(), lr=0.01, momentum=0.9)
        delta_label = None

        for epoch in range(epochs):
            with torch.no_grad():
                self.dec_model=self.dec_model.to("cpu")## predict full dataset should not on CUDA,but on cpu
                q=self.dec_model(dataset.tensors[0]).cpu()
                p=target_distribution(q).detach()
                y_pred=q.max(1)[1].cpu().data.numpy()   
                dt_set= torch.utils.data.TensorDataset(torch.FloatTensor(self.x),p)
                dataloader=DataLoader(dt_set,batch_size=batch_size,shuffle=True)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if epoch>0 and stopping_delta is not None and delta_label < stopping_delta:
                    print(
                        'Early stopping as label delta "%1.5f" less than "%1.5f".'
                        % (delta_label, stopping_delta)
                    )
                    break
            self.dec_model=self.dec_model.to(device)
            self.dec_model.train()
            
            print("training with epochs_fit......")
            for i in range(epochs_fit):
                print("epoch_fit={}".format(i+1))
                for index,(input,p_prob) in enumerate(dataloader):
                    p_prob=p_prob.to(device)
                    output = self.dec_model(input.to(device))
                    loss = loss_function(output.log(), p_prob) / output.shape[0]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step(closure=None)


    # def dec_train(
    #         self,
    #         init_pred:None,
    #         cluster_centers:None,
    #         dataset: torch.utils.data.Dataset,
    #         epochs: int,
    #         batch_size: int,
    #         optimizer: Optional[torch.optim.Optimizer]=None,
    #         stopping_delta: Optional[float] = None,
    #         sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    #         silent: bool = False,
    #         update_freq: int = 10,
    #         evaluate_batch_size: int = 1024,
    #         update_callback: Optional[Callable[[float, float], None]] = None,
    #         epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    # ) -> None:
    #     """
    #     Train the DEC model given a dataset, a model instance and various configuration parameters.

    #     :param dataset: instance of Dataset to use for training
    #     :param model: instance of DEC model to train
    #     :param epochs: number of training epochs
    #     :param batch_size: size of the batch to train with
    #     :param optimizer: instance of optimizer to use
    #     :param stopping_delta: label delta as a proportion to use for stopping, None to disable, default None
    #     :param collate_fn: function to merge a list of samples into mini-batch
    #     :param cuda: whether to use CUDA, defaults to True
    #     :param sampler: optional sampler to use in the DataLoader, defaults to None
    #     :param silent: set to True to prevent printing out summary statistics, defaults to False
    #     :param update_freq: frequency of batches with which to update counter, None disables, default 10
    #     :param evaluate_batch_size: batch size for evaluation stage, default 1024
    #     :param update_callback: optional function of accuracy and loss to update, default None
    #     :param epoch_callback: optional function of epoch and model, default None
    #     :return: None
    #     """
    #     print("dec training......")
    #     static_dataloader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         pin_memory=False,
    #         shuffle=False,
    #     )
    #     train_dataloader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #     )
    #     data_iterator = tqdm(
    #         static_dataloader,
    #         leave=True,
    #         unit="batch",
    #         postfix={
    #             "epo": -1,
    #             "train_loss": "%.8f" % 0.0,
    #         }
    #     )

    #     predicted=init_pred.copy()
    #     self.dec_model.train()

    #     predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)

    #     cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True)

    #     with torch.no_grad():
    #         # initialise the cluster centers
    #         self.dec_model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    #     loss_function = nn.KLDivLoss(size_average=False)
    #     optimizer = SGD(self.dec_model.parameters(), lr=0.01, momentum=0.9)
    #     delta_label = None
    #     for epoch in range(epochs):
    #         features = []
    #         self.dec_model.train()
    #         for i in range(self.epochs_fit):
    #             data_iterator = tqdm(
    #                 train_dataloader,
    #                 leave=True,
    #                 unit="batch",
    #                 postfix={
    #                     "epo": epoch,
    #                     "train_loss": "%.8f" % 0.0,
    #                 },
    #                 disable=silent,
    #             )
    #             for index, batch in enumerate(data_iterator):
    #                 if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
    #                         batch
    #                 ) == 2:
    #                     batch, _ = batch  # if we have a prediction label, strip it away

    #                 output = self.dec_model(batch[0])
    #                 target = target_distribution(output).detach()
    #                 loss = loss_function(output.log(), target) / output.shape[0]
    #                 data_iterator.set_postfix(
    #                     epo=epoch,
    #                     train_loss="%.8f" % float(loss.item()),
    #                 )
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step(closure=None)
    #                 features.append(self.dec_model.encoder(batch[0]).detach().cpu())

    #         predicted = self.predict(
    #             dataset,
    #             self.dec_model,
    #             batch_size=self.batch_size,
    #             return_actual=False
    #         )
    #         delta_label = (
    #                 float((predicted != predicted_previous).float().sum().item())
    #                 / predicted_previous.shape[0]
    #         )
    #         print("The value of delta_label of current", str(epoch + 1), "th iteration is", delta_label, ">= tol",
    #               stopping_delta)
    #         if stopping_delta is not None and delta_label < stopping_delta:
    #             print(
    #                 'Early stopping as label delta "%1.5f" less than "%1.5f".'
    #                 % (delta_label, stopping_delta)
    #             )
    #             break
    #         predicted_previous = predicted
    #         data_iterator.set_postfix(
    #             epo=epoch,
    #             train_loss="%.8f" % 0.0,
    #         )

    def predict(self,
            dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            batch_size: int = 1024,
            return_actual: bool = False,
            return_prob=False,
    ) :
        """
        Predict clusters for a dataset given a DEC model instance and various configuration parameters.

        :param dataset: instance of Dataset to use for training
        :param model: instance of DEC model to predict
        :param batch_size: size of the batch to predict with, default 1024
        :param return_actual: return actual values, if present in the Dataset
        :return: tuple of prediction and actual if return_actual is True otherwise prediction
        """
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=True, unit="batch")
        features = []
        actual = []
        model.eval()
        for batch in data_iterator:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # unpack if we have a prediction label
                if return_actual:
                    actual.append(value)
            elif return_actual:
                raise ValueError(
                    "Dataset has no actual value to unpack, but return_actual is set."
                )
            features.append(
                model(batch[0]).detach().cpu()
            )  # move to the CPU to prevent out of memory on the GPU

        if return_prob:
            return torch.cat(features)
        else:
            return torch.cat(features).max(1)[1]


    def exact_features(self,
            dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            batch_size: int,
            encode=True,
            device=torch.device("cpu")
    ) -> torch.Tensor:
        """
        Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
        output.

        :param dataset: evaluation Dataset
        :param model: autoencoder for prediction
        :param batch_size: batch size
        :param cuda: whether CUDA is used, defaults to True
        :param silent: set to True to prevent printing out summary statistics, defaults to False
        :param encode: whether to encode or use the full autoencoder
        :return: predicted features from the Dataset
        """
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=False, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
        features = []
        if isinstance(model, torch.nn.Module):
            model.eval()
        for batch in data_iterator:
            if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
                batch = batch[0]

            batch = batch.squeeze(1).view(batch.size(0), -1)
            batch = batch.to(device)
            if encode:
                output = model.encode(batch)
            else:
                output = model(batch)
            features.append(
                output.detach().cpu()
            )  # move to the CPU to prevent out of memory on the GPU
        return torch.cat(features)
        
    def get_embedding(self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        batch_size: int = 1024,
        return_actual: bool = False,
    ):
        """
        Predict clusters for a dataset given a DEC model instance and various configuration parameters.

        :param dataset: instance of Dataset to use for training
        :param model: instance of DEC model to predict
        :param batch_size: size of the batch to predict with, default 1024
        :param return_actual: return actual values, if present in the Dataset
        :return: tuple of prediction and actual if return_actual is True otherwise prediction
        """
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=True, unit="batch")
        features = []
        actual = []
        model.eval()
        for batch in data_iterator:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # unpack if we have a prediction label
                if return_actual:
                    actual.append(value)
            elif return_actual:
                raise ValueError(
                    "Dataset has no actual value to unpack, but return_actual is set."
                )
            features.append(
                model.encoder(batch[0]).detach().cpu()
            )  # move to the CPU to prevent out of memory on the GPU
        embedding=torch.cat(features).numpy()
        print(embedding.shape)
        return torch.cat(features).data.numpy()


    
    
