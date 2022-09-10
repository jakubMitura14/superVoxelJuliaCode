"""
Calculates weather the variance of the features is related to variance of the supervoxel probability
basically if features are relatively constant so some are looks mostly the same the probability
    of supervoxel presence shuld be relatively constant - hovewer spots where variance of features is high it indicates the boundry between 
    supervoxels
"""


Flux.Losses.logitbinarycrossentropy(a,b)
