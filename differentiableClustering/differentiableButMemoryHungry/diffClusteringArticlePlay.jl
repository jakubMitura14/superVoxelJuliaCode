"""
implementatin of paper https://arxiv.org/pdf/2007.09990.pdf
and adaptation from https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/
summarizing article as far as I get It 

I) feature extraction and batch normalization
II) 1D convolution in space of dimensions equal to maximum numbers of clusters
III) batch normalization
IV) spatial and feature similarity loss

problem is that it basically requires to add as many channels as there are possible supervoxels
idea that can be tried is to  do all per voxel in the Enzyme
where channels will be hold in the shared memory
so 1) normal set of convolution to get deep features
then in enzyme we get single voxel - spread it using big convolution
    basically it seem to be quite sililar to dense clusterLayer
    and spread it so spreaded - increased channel representation will be hold in shared memory Array
    we can then normalize it in this shared memory 
    and calculate statistics required for losses - frankly atomic add would be extremely usefull
        
"""

