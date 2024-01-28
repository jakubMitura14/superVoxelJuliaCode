using Revise, Test, Plots, Random
includet("/workspaces/superVoxelJuliaCode/sequentialMultiLayer/utilsSequential/includeAllSequential.jl")
using Pkg
Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")
"""
testing the model whheather it compiles
"""

rng = Random.MersenneTwister()
Nx, Ny, Nz=64,64,64
oneSidePad=0
crossBorderWhere=32
image,indArr=createTestDataFor_Clustering(Nx, Ny, Nz, oneSidePad, crossBorderWhere)
r_features=4
r_feature_variance=3
featuresNumb=2

threads_CalculateFeatures=(8,8,8) 
blocks_CalculateFeatures=(8,8,8)
threads_CalculateFeatures_variance=(8,8,8)
blocks_CalculateFeatures_variance=(8,8,8)


image_withFeatures=prepareFeatures( CuArray(image)
    ,r_features
    ,r_feature_variance
    ,featuresNumb
    ,threads_CalculateFeatures,blocks_CalculateFeatures
    ,threads_CalculateFeatures_variance,blocks_CalculateFeatures_variance )

### define model layers
numberOfConv2=4


threads_spreadKern=(8,8,8)
blocks_spreadKern=(8,8,8)
threads_featureLoss_kern_=(8,8,8)
blocks_featureLoss_kern_=(8,8,8)
threads_disagreeKern=(8,8,8)
blocks_disagreeKern=(8,8,8)
threads_blockss=threads_blocks_struct(threads_spreadKern,blocks_spreadKern,threads_featureLoss_kern_, blocks_featureLoss_kern_,threads_disagreeKern,blocks_disagreeKern)

supervoxel_numb=8

dim_x,dim_y,dim_z=Nx, Ny, Nz

model = getModel(numberOfConv2,Nx, Ny, Nz, featuresNumb, supervoxel_numb,threads_blockss::threads_blocks_struct )
ps, st = Lux.setup(rng, model)
opt = Optimisers.NAdam(0.003)


function clusteringLossA(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    # print("   sizzzz $(size(y_pred))       ")
    res= y_pred[1][2]
    return res, st, ()

    # return 1*(sum(y_pred)), st, ()
end


# tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
#tstate = Lux.Training.TrainState(rng, model, opt)
vjp_rule = Lux.Training.ZygoteVJP()

function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data,
    epochs::Int)
   # data = data .|> Lux.gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, clusteringLossA,
                                                                data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

tstate = main(tstate, vjp_rule, image_withFeatures,1)






# reductionFactor=2^numberOfConv2
# rdim_x,rdim_y,rdim_z=Int(round(dim_x/reductionFactor )),Int(round(dim_y/reductionFactor )),Int(round(dim_z/reductionFactor ))
