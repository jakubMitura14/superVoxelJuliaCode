

using MedPipe3D
using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils, MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved
# using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA, HDF5, Colors, ParallelStencil, ParallelStencil.FiniteDifferences3D
using MedEval3D, MedEval3D.BasicStructs, MedEval3D.MainAbstractions
using MedEval3D, MedEval3D.BasicStructs, MedEval3D.MainAbstractions, Hyperopt, Plots
using MedPipe3D.LoadFromMonai
import Lux
import NNlib, Optimisers, Plots, Random, Statistics, Zygote, HDF5
using PythonCall


# # some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# # 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl
# dim_x,dim_y,dim_z=32,32,32


# rng = Random.default_rng()
conv1 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.tanh, stride=1, pad=Lux.SamePad())

function getConvModel()
    return Lux.Chain(conv1(1, 4), conv1(4, 8), conv1(8, 16), conv1(16, 8), conv1(8, 4), conv1(4, 2), conv1(2, 1))
end#getConvModel




# # """
# # good tutorial below
# # https://gowrishankar.info/blog/calculus-gradient-descent-optimization-through-jacobian-matrix-for-a-gaussian-distribution/
# # generally estimating mean and variance of gaussian distributions needs to be part of the bincludet("/media/jakub/NewVolume/projects/superVoxelJuliaCode/utils/includeAll.jl")
# using Distributions

# Nx, Ny, Nz = 32, 32, 32
# oneSidePad = 1
# totalPad = oneSidePad*2
# dim_x,dim_y,dim_z= Nx+totalPad, Ny+totalPad, Nz+totalPad

# crossBorderWhere = 16
# # sitk=MedPipe3D.LoadFromMonai.getSimpleItkObject()
# pathToHDF5="/home/jakub/CTORGmini/smallDataSet.hdf5"
# data_dir = "/home/jakub/CTORGmini"

# #how many gaussians we will specify 
# const gauss_numb_top = 8
# threads_apply_gauss = (8, 8, 8)
# blocks_apply_gauss = (4, 4, 4)



# rng = Random.default_rng()
# origArr,indArr=createTestDataFor_Clustering(Nx, Ny, Nz, oneSidePad, crossBorderWhere)


# modelConv = getConvModel()
# gaussApplyLayer=Gauss_apply(gauss_numb_top,threads_apply_gauss,blocks_apply_gauss)
# model=Lux.Chain(
#                     modelConv,
#                     gaussApplyLayer
#                     )
# ps, st = Lux.setup(rng, model)
# x =reshape(origArr, (dim_x,dim_y,dim_z,1,1))
# x= CuArray(x)
# #y_pred, st =Lux.apply(model, x, ps, st) 

# opt = Optimisers.NAdam(0.003)
# #opt = Optimisers.Adam(0.003)
# #opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0), Optimisers.NAdam());







# const tops=1+Int(oneSidePad)
# const tope=Int(crossBorderWhere)+ Int(oneSidePad)
# const bottoms=crossBorderWhere
# const bottome=Nz+ oneSidePad
# const lefts=1+oneSidePad
# const lefte=crossBorderWhere+ oneSidePad
# const rights=crossBorderWhere+ oneSidePad+1
# const righte=Nx+ oneSidePad
# const anteriors=crossBorderWhere+ oneSidePad+1
# const anteriore=Ny+ oneSidePad
# const posteriors=1+oneSidePad
# const posteriore=crossBorderWhere+ oneSidePad


# function loss_function(model, ps, st, x)
#     y_pred, st = Lux.apply(model, x, ps, st)

#     top_left_post =view(y_pred,tops:tope,lefts:lefte, posteriors:posteriore )
#     top_right_post =view(y_pred,tops:tope,rights:righte, posteriors:posteriore)

#     top_left_ant =view(y_pred,tops:tope,lefts:lefte, anteriors:anteriore )
#     top_right_ant =view(y_pred,tops:tope,rights:righte, anteriors:anteriore ) 

#     bottom_left_post =view(y_pred,bottoms:bottome,lefts:lefte, posteriors:posteriore ) 
#     bottom_right_post =view(y_pred,bottoms:bottome,rights:righte, posteriors:posteriore ) 

#     bottom_left_ant =view(y_pred,bottoms:bottome,lefts:lefte, anteriors:anteriore )
#     bottom_right_ant =view(y_pred,bottoms:bottome,rights:righte, anteriors:anteriore ) 

#     varss= map(var  ,[top_left_post,top_right_post, top_left_ant,top_right_ant,bottom_left_post,bottom_right_post,bottom_left_ant, bottom_right_ant ])
#     means= map(mean  ,[top_left_post,top_right_post, top_left_ant,top_right_ant,bottom_left_post,bottom_right_post,bottom_left_ant, bottom_right_ant ])


#     # so we want maximize the ypred values so evrywhere we will have high prob in some gaussian
#     # minimize variance inside the regions
#     # maximize variance between regions
#     res= sum(varss)-sum(y_pred) -var(means)
#     return res, st, ()

#     # return 1*(sum(y_pred)), st, ()
# end


# # tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
# tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
# #tstate = Lux.Training.TrainState(rng, model, opt)
# vjp_rule = Lux.Training.ZygoteVJP()


# function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data,
#     epochs::Int)
#    # data = data .|> Lux.gpu
#     for epoch in 1:epochs
#         grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
#                                                                 data, tstate)
#         @info epoch=epoch loss=loss
#         tstate = Lux.Training.apply_gradients(tstate, grads)
#     end
#     return tstate
# end

# # x = randn(rng, Float32, dim_x,dim_y,dim_z)
# # x =reshape(origArr, (dim_x,dim_y,dim_z,1,1))
# # tstate = main(tstate, vjp_rule, CuArray(x),1)
# origArr=CuArray(x)
# tstate = main(tstate, vjp_rule, origArr,1500)


# ############################ visualization


# function applyGaussKern_for_vis(means,stdGaus,origArr,out,meansLength)
#     #adding one becouse of padding
#     x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
#     y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
#     z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
#     #iterate over all gauss parameters
#     maxx = 0.0
#     index=0

#     for i in 1:meansLength
#        vall=univariate_normal(origArr[x,y,z,1,1], means[i], stdGaus[i]^2)
#        CUDA.@cuprint "vall $(vall) i $(i)   " 
#        if(vall>maxx)
#             maxx=vall
#             index=i
#         end #if     
#     end #for

#     out[x,y,z,1,1]=float(index)    
#     return nothing
# end
# psss=tstate.parameters
# l1,l2,l3,l4,l5,l6=psss
# stdGaus,means=l6
# out = CUDA.zeros(size(origArr))
# ence need to be 
# # done via optimazation based on idea on gradient descent - challenge here is to keep the SPD structure of the covariance matrix
# #     simple way to avoid it is to just train univariate distributions - and hopfully the other layesr will lead to the intensity values that are uniform enough 
# # """
# # function optimize_gauss_backprop()

# # end

# # """
# # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# # https://leenashekhar.github.io/2019-01-30-KL-Divergence/
# # """
# # function klDiv_univariate_gaussians()
# #     KL(p,q)=logσ2σ1+σ21+(μ1−μ2)22σ22−12
# # end    


# # """
# # https://sunil-s.github.io/assets/pdfs/multivariate_mutual_information.pdf
# # """
# # function mutualInformation()


# # end#mutualInformation







# # #modelConv=Lux.Chain(conv1(1,4),conv1(4,16),conv1(16,4),conv1(4,1))
# # ps, st = Lux.setup(rng, modelConv)
# # x = randn(rng, Float32, dim_x,dim_y,dim_z)
# # x =reshape(x, (dim_x,dim_y,dim_z,1,1))
# # y_pred, st =Lux.apply(modelConv, x, ps, st) 
# # size(y_pred)

# # """
# # just bunch of 3d convolutions together they should create/modify either the edges or the regions depending on interpretation
# # """
# # function getGenericconv()

# # end#getGenericconv