DESC--Deep Embedded Single-cell RNA-seq Clustering implementation with pytorch
===============================================================================
DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data.
The algorithm constructs a non-linear mapping function from the original scRNA-seq data 
space to a low-dimensional feature space by iteratively learning cluster-specific gene expression 
representation and cluster assignment based on a deep neural network.
 This iterative procedure moves each cell to its nearest cluster, 
 balances biological and technical differences between clusters, 
 and reduces the influence of batch effect.
 DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, 
 which facilitates the identification of cells clustered with high-confidence and interpretation of results.
 the implementation of tensorflow version can be found https://github.com/eleozzr/desc.
 this project is the implementation of DESC by pytorch. If you have any question, you can contact us with 
 yuxiaokang12@gmail.com and ele717@163.com.