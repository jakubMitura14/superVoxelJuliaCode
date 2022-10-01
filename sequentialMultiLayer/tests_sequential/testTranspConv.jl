using Revise, Test, Plots, Random
includet("/workspaces/superVoxelJuliaCode/sequentialMultiLayer/utilsSequential/includeAllSequential.jl")
using Pkg
# Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")
includet("/workspaces/superVoxelJuliaCode/sequentialMultiLayer/pureEnzyme/customTransConv.jl")



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


featureNumb=2
contractPart = getContractModel(featureNumb+1,2)

blocks_transConv_kernel=(16,16,16)
threads_transConv_kernel=(4,4,4)
numParams=2


model=Lux.Chain(contractPart
,transConv_layer(numParams,threads_transConv_kernel,blocks_transConv_kernel )
,transConv_layer(numParams,threads_transConv_kernel,blocks_transConv_kernel )
,transConv_layer(numParams,threads_transConv_kernel,blocks_transConv_kernel )
,transConv_layer(numParams,threads_transConv_kernel,blocks_transConv_kernel )
)


ps, st = Lux.setup(rng, model)
opt = Optimisers.NAdam(0.003)


function clusteringLossA(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    print(" gettting losss y_pred $(size(y_pred[:,:,:,1,1]))  x $(size(x[:,:,:,1,1]))  ")
    # return 0.1
    res= sum((y_pred[:,:,:,1,1]-x[:,:,:,1,1]).^2)
    print(" resss $(res) ")
    return res,st, ()

end


tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
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

tstate = main(tstate  , vjp_rule, image_withFeatures,1)


Lux.gpu(ps)

res=model(image_withFeatures,Lux.gpu(ps), Lux.gpu(st))

ps['layer_12']



