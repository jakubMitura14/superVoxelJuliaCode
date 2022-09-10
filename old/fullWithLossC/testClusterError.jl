import Lux,CUDA
import NNlib, Optimisers, Plots, Random, Statistics, Zygote, HDF5
using CUDA

Nx, Ny, Nz = 32, 32, 32
oneSidePad = 1
totalPad = oneSidePad*2
dim_x,dim_y,dim_z= Nx+totalPad, Ny+totalPad, Nz+totalPad
featureNumb=3
conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())
rng = Random.default_rng()

function myCatt(a,b)
    cat(a,b;dims=4)
end    

modelConv=Lux.Chain(conv1(featureNumb,4),conv1(4,16),conv1(16,4),conv1(4,3))
modelConv=Lux.SkipConnection(modelConv,myCatt)
# modelConv=Lux.BranchLayer(modelConv,Lux.NoOpLayer)
ps, st = Lux.setup(rng, modelConv)
x = ones(rng, Float32, dim_x,dim_y,dim_z,featureNumb)
x =CuArray(reshape(x, (dim_x,dim_y,dim_z,featureNumb,1)))
# y_pred, st =Lux.apply(modelConv, x, ps, st) 
# size(y_pred)


model=modelConv
opt = Optimisers.NAdam(0.003)

function clusteringLoss(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    print("   sizzzz $(size(y_pred))       ")
    res= sum(y_pred)
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
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, clusteringLoss,
                                                                data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end


tstate = main(tstate, vjp_rule, CuArray(x),1)

