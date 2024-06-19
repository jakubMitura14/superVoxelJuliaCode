using Revise, CUDA,HDF5
using Meshes
using LinearAlgebra
using GLMakie
using Combinatorics
using SplitApplyCombine
using CUDA
using Combinatorics
using Random
using Statistics
using ChainRulesCore
using Test
using ChainRulesTestUtils
using EnzymeTestUtils
using Logging, FiniteDifferences, FiniteDiff
using Interpolations
using KernelAbstractions,Dates
# using KernelGradients
using Zygote, Lux, LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra

using Revise

includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/Lux_model.jl")



h5_path="/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/data/synth_data.h5"
f = h5open(h5_path, "r")

rng = Random.default_rng()
threads = (2, 2, 2)
blocks = (1, 1, 1)
dev = gpu_device()


function get_sample_dat(f::HDF5.File)

    return f["1/image"][:,:,:],f["1/weights"][:,:,:,:],f["1/tetr"][:,:,:]
    # return f["1/image"][:,:,:],f["1/weights"][:,:,:,:],f["1/tetr"][:,:,:,:]
end

imagee,weights,tetr_out_saved = get_sample_dat(f)
imagee=reshape(imagee, (size(imagee)[1],size(imagee)[2],size(imagee)[3],1,1))

radiuss = Float32(4.0)
image_shape=size(imagee)
# threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(size(control_points), size(weights))


#get convolutions
conv1 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.tanh, stride=1, pad=Lux.SamePad(),init_weight=glorot_uniform)
conv2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.tanh, stride=2, pad=Lux.SamePad(),init_weight=glorot_uniform)
convsigm2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=2, pad=Lux.SamePad())


# Conv(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer},
#      activation=identity; init_weight=glorot_uniform, init_bias=zeros32, stride=1,
#      pad=0, dilation=1, groups=1, use_bias=true, allow_fast_activation=true)


weights_dims=(get_corrected_dim(1,radiuss,image_shape)
        ,get_corrected_dim(2,radiuss,image_shape)
        ,get_corrected_dim(3,radiuss,image_shape))

function connection_before_kernelA(x, y)
    return (x, y)
end

#conv part is to get the weights of write size (in case of radiuss 4 we need 3 times stride 2 convolutions
# to get the weights fitted to number of super voxels and we need 6 channels as it is number of weights per super voxel)
num_convs_per_dim=(3,3,3)
conv_part=Lux.Chain(conv2(1, 6), conv2(6, 6),conv2(6, 6))


"""
conv_part give us weights in semi correct shape and corect number of channels
now we need to have the skip connection to the point kernels as those require original image for sampling
"""
model = Lux.Chain(conv_part,Points_weights_str(radiuss,(image_shape[1],image_shape[2],image_shape[3]),num_convs_per_dim))
# model = Lux.Chain(SkipConnection(Lux.Chain(conv1(1, 3), conv2(3, 3), convsigm2(3, 3))
#         , connection_before_kernelA; name="prim_convs")
#         , KernelA(Nx, threads, blocks))
opt = Optimisers.Adam(0.003)
ps, st = Lux.setup(rng, model)
st=cu(st)
ps=cu(ps)
y_pred, st = Lux.apply(model, CuArray(imagee), ps, st)

sum(y_pred)


size(y_pred)
a
# function loss_function(model, ps, st, x)
#     y_pred, st = Lux.apply(model, x, ps, st)
#     return (sum(y_pred)), st, ()
# end


# function main(ps, st, opt, opt_st, vjp, data, model,
#     epochs::Int)
#     x = CuArray(data) #.|> Lux.gpu
#     for epoch in 1:epochs

#         (loss, st), back = Zygote.pullback(p -> loss_function(model, p, st, x), ps)
#         gs = back((one(loss), nothing))[1]
#         opt_st, ps = Optimisers.update(opt_st, ps, gs)

#         # grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
#         #                                                         data, tstate)
#         @info epoch = epoch loss = loss #tstate=tstate.parameters.paramsA
#         # tstate = Lux.Training.apply_gradients(tstate, grads)
#     end
#     return ps, st, opt, opt_st
# end

# #initialization
# model, ps, st, opt, opt_st, vjp_rule = get_model_consts(dev, Nx, threads, blocks)

# # y_pred, st = Lux.apply(model, x, ps, st)

# # one epoch just to check if it runs
# ps, st, opt, opt_st = main(ps, st, opt, opt_st, vjp_rule, image, model, 1)

#training 
# ps, st, opt, opt_st = main(ps, st, opt, opt_st, vjp_rule, x, model, 300)

