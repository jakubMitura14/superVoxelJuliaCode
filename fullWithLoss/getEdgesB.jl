

using MedPipe3D
using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved
# using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA,HDF5,Colors,ParallelStencil, ParallelStencil.FiniteDifferences3D
using MedEval3D, MedEval3D.BasicStructs, MedEval3D.MainAbstractions
using MedEval3D, MedEval3D.BasicStructs, MedEval3D.MainAbstractions,Hyperopt,Plots
using MedPipe3D.LoadFromMonai
import Lux
import NNlib, Optimisers, Plots, Random, Statistics, Zygote, HDF5

# Nx, Ny, Nz = 32, 32, 32
# oneSidePad = 1
# totalPad = oneSidePad*2
# dim_x,dim_y,dim_z= Nx+totalPad, Ny+totalPad, Nz+totalPad
# featureNumb=3
# conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())
# rng = Random.default_rng()

# function myCatt(a,b)
#     cat(a,b;dims=4)
# end    

# modelConv=Lux.Chain(conv1(featureNumb,4),conv1(4,16),conv1(16,4),conv1(4,3))
# modelConv=Lux.SkipConnection(modelConv,myCatt)
# ps, st = Lux.setup(rng, modelConv)
# x = ones(rng, Float32, dim_x,dim_y,dim_z,featureNumb)
# x =reshape(x, (dim_x,dim_y,dim_z,featureNumb,1))
# y_pred, st =Lux.apply(modelConv, x, ps, st) 
# size(y_pred)
# y_pred[1,1,1,4,1]



# # some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# # 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl
# dim_x,dim_y,dim_z=32,32,32


# rng = Random.default_rng()
conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())

function getConvModel()
    return Lux.Chain(conv1(3,4),conv1(4,8),conv1(8,16),conv1(16,8),conv1(8,4),conv1(4,2),conv1(2,1))
end#getConvModel



