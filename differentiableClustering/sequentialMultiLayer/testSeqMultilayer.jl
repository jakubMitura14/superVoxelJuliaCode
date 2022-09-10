
"""
testing the model whheather it compiles
"""
rng = Random.MersenneTwister()
dim_x,dim_y,dim_z=64,64,64
base_arr=rand(dim_x,dim_y,dim_z )
base_arr=Float32.(reshape(base_arr, (dim_x,dim_y,dim_z,1,1)))




ps, st = Lux.setup(rng, model)
out = Lux.apply(model, base_arr, ps, st)
size(out[1])

######## define features array
image=base_arr
r_features=2
r_feature_variance=1
featuresNumb=2

image_withFeature_var=prepareFeatures( image
    ,r_features
    ,r_feature_variance
    ,featuresNumb
    ,threads_CalculateFeatures,blocks_CalculateFeatures
    ,threads_CalculateFeatures_variance,blocks_CalculateFeatures_variance )

### define model layers
numberOfConv2=4