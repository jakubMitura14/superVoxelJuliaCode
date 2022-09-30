

"""
geting the reduced representation after convolutions 
and the original array with extracted features to establish losses
1) get dense matrix multiplied reduced representation to keep information about location of supervoxel
2) do the transposed convolutions (simplified manual version) until get the original image size 
3) do soft thresholding to get the semithresholded vesrion - close to 1-s where the supervoxel is present and close to zeros if distant
4) calculate losses
    establish spread as variance of x,y,z s in high value p locations
    check the variance of features in high values region - the lower variance the blocks_CalculateFeatures_variance
    if it is next supervoxel establish weather there is overlap in high values - for example multiply add and raise to power - the bigger the value the worse                
"""
function getSuperVoxels(origArrWithFeatures,reducedRepresentation, feature_number)
    


end #getSuperVoxels    


"""
forward definition of the supervoxel
"""
function getSuperVoxelForward(origArrWithFeatures,reducedRepresentation, feature_number,orig3DIms,reduced3Dims)
    
    Aout=CUDA.zeros(reduced3Dims)
    elementWise_kernel(source,toMultiplyWith,Aout,Nx) # element wise multiplication - keeping info about location
    

end#getSuperVoxelForward    