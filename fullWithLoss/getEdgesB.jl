

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
using PythonCall


# # some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# # 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl
# dim_x,dim_y,dim_z=32,32,32


# rng = Random.default_rng()
conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())

function getConvModel()
    return Lux.Chain(conv1(1,4),conv1(4,8),conv1(8,4),conv1(4,2),conv1(2,1))
end#getConvModel




# """
# good tutorial below
# https://gowrishankar.info/blog/calculus-gradient-descent-optimization-through-jacobian-matrix-for-a-gaussian-distribution/
# generally estimating mean and variance of gaussian distributions needs to be part of the backpropagation chain hence need to be 
# done via optimazation based on idea on gradient descent - challenge here is to keep the SPD structure of the covariance matrix
#     simple way to avoid it is to just train univariate distributions - and hopfully the other layesr will lead to the intensity values that are uniform enough 
# """
# function optimize_gauss_backprop()

# end

# """
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://leenashekhar.github.io/2019-01-30-KL-Divergence/
# """
# function klDiv_univariate_gaussians()
#     KL(p,q)=logσ2σ1+σ21+(μ1−μ2)22σ22−12
# end    


# """
# https://sunil-s.github.io/assets/pdfs/multivariate_mutual_information.pdf
# """
# function mutualInformation()


# end#mutualInformation







# #modelConv=Lux.Chain(conv1(1,4),conv1(4,16),conv1(16,4),conv1(4,1))
# ps, st = Lux.setup(rng, modelConv)
# x = randn(rng, Float32, dim_x,dim_y,dim_z)
# x =reshape(x, (dim_x,dim_y,dim_z,1,1))
# y_pred, st =Lux.apply(modelConv, x, ps, st) 
# size(y_pred)

# """
# just bunch of 3d convolutions together they should create/modify either the edges or the regions depending on interpretation
# """
# function getGenericconv()

# end#getGenericconv