using Flux,Lux, Random,Optimisers,Revise, CUDA
# include("C:\\projects\\superVoxelJuliaCode\\differentiableClustering\\sequentialMultiLayer\\unetLux.jl")
# include("C:\\projects\\superVoxelJuliaCode\\differentiableClustering\\sequentialMultiLayer\\multLayer.jl")




"""
calculates features of the image and then the local variance of those features
it leaves original image as the first channel of created array
image - image in which we want to calculate features - by convention it should be 5 dimensional 4th is channels 5th is batch
r_features - the radius which will be used to calculate features
r_feature_variance - the radius to define local feature variance
featuresNumb - how many feature we defined to be analyzed
threads.. , blocks... - definitions for the CUDA needed to know how many threads
and blocks should be run
"""
function prepareFeatures( image
    ,r_features
    ,r_feature_variance
    ,featuresNumb
    ,threads_CalculateFeatures,blocks_CalculateFeatures
    ,threads_CalculateFeatures_variance,blocks_CalculateFeatures_variance )
    mainArrSize= size(image)
    featuresArrSize=(mainArrSize...,featuresNumb+1,1 )
     image_features=call_calculateFeatures(image,mainArrSize,r_features,featuresNumb,threads_CalculateFeatures,blocks_CalculateFeatures )

    return call_calculateFeatures_varianvce(image_features,featuresArrSize,r_feature_variance,featuresNumb
    ,threads_CalculateFeatures_variance,blocks_CalculateFeatures_variance )
end#prepareFeatures






