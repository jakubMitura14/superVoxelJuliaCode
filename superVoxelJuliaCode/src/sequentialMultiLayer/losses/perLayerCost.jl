"""
each supervoxel layer should also establish its own cost based on spread - penalizing
if high probability voxels are spread wide apart and weather high probability means also high similarity
in feature space
1) normalize the calculated probabilities keep it between 0 and 1 we can use LuX normalize 
2) get variance of location - so for each point multiply its x,y,z cooridinates by the value at the st_opt
    save it in appropriate arrays avoiding data races
    and return means of them as proposed barycenter
    then repeat operation but this time given means caclulate variance, the higher variance the more spread 
    are the points
    alternatively a bit simpler maybe we can calculate the variance of current position relative to 3
    corners (triangulation)
    so we get euclidean distance to corner a times p we save then the same relative to corner b and C
    finally from saved 3 arrays we calculate variances and add them up  
3)in case of features we should try sth like in [1] contrastive regional loss but we will start from simpler case
    so we can say that variance of p should behave similarly as variance of feature values 
    so areas where feature variance is low - meaning area looks the same variance of p should be small
    in areas of high variance of feature the p should also change as we are transitioning to other area
    hovewer we will also additionally multiply by current value of the p becouse we mostly care about analyzing
    areas that are in this super voxel and not background
    we can probably prepare the varianceof features beforehand - before Training loop
    and variance of p we we will calculate here 
    next we will evaluate  Flux.Losses.logitbinarycrossentropy between var p and var f1 plus Flux.Losses.logitbinarycrossentropy 
    between var p and varf2 ...
    
1 https://arxiv.org/pdf/2104.04465.pdf    
"""






