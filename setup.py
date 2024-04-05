from setuptools import setup, find_packages

setup(name = 'DESCtorch',
			version = '1.1.0',
			description = 'Deep Embedded Single-cell RNA-seq Clustering implementation with pytorch',
			long_description = 'DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm constructs a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learning cluster-specific gene expression representation and cluster assignment based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances biological and technical differences between clusters, and reduces the influence of batch effect. DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, which facilitates the identification of cells clustered with high-confidence and interpretation of results.',
			classifiers = [
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8',
      	],
			url = 'https://github.com/yuxiaokang-source/DESCtorch',
			author = 'Xiaokang Yu,Xiangjie Li',
			author_email = 'yuxiaokang2018@163.com,ele717@163.com',
			license = 'MIT',
			packages = find_packages(),
			include_package_data=True,
			install_requires = [
				'torch==1.10.2',
				'scanpy==1.9.2',
				'louvain==0.8.0',
				'cytoolz==0.11.2'
				],
			zip_safe = False
                        )
