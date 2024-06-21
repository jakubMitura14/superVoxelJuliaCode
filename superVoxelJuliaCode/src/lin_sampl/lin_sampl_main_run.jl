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
using Lux, Random, Optimisers, Zygote
using LinearAlgebra

using Revise



includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/Lux_model.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_variance_for_loss.jl")



h5_path="/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/data/synth_data.h5"
f = h5open(h5_path, "r")

rng = Random.default_rng()
# dev = gpu_device()
num_convs_per_dim=(3,3,3)

function get_sample_dat(f::HDF5.File)

    return f["1/image"][:,:,:],f["1/weights"][:,:,:,:],f["1/tetr"][:,:,:]
    # return f["1/image"][:,:,:],f["1/weights"][:,:,:,:],f["1/tetr"][:,:,:,:]
end



imagee,weights,tetr_out_saved = get_sample_dat(f)
imagee=reshape(imagee, (size(imagee)[1],size(imagee)[2],size(imagee)[3],1,1))




radiuss = Float32(4.0)
pad_voxels=2
num_base_samp_points,num_additional_samp_points=3,2
# imagee=pad_source_arr(imagee,pad_voxels)
image_shape=size(imagee)
# threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(size(control_points), size(weights))


#get convolutions
conv1 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.tanh, stride=1, pad=Lux.SamePad(),init_weight=glorot_uniform)
conv2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.tanh, stride=2, pad=Lux.SamePad(),init_weight=glorot_uniform)
convsigm2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=2, pad=Lux.SamePad())
function connection_before_set_tetr_dat_kern(x, y)
    return (x, y)
end
#conv part is to get the weights of write size (in case of radiuss 4 we need 3 times stride 2 convolutions
# to get the weights fitted to number of super voxels and we need 6 channels as it is number of weights per super voxel)
conv_part=Lux.Chain(conv2(1, 10), conv2(10, 12),conv2(12, 6))


"""
conv_part give us weights in semi correct shape and corect number of channels
now we need to have the skip connection to the point kernels as those require original image for sampling
"""
before_point_kerns = SkipConnection(Lux.Chain(conv_part
,Points_weights_str(radiuss,pad_voxels,(image_shape[1],image_shape[2],image_shape[3]),num_convs_per_dim)),connection_before_set_tetr_dat_kern)
model = Lux.Chain(before_point_kerns
                ,Set_tetr_dat_str(radiuss,pad_voxels,(image_shape[1],image_shape[2],image_shape[3]))
                ,Point_info_kern_str(radiuss,(image_shape[1],image_shape[2],image_shape[3]),num_base_samp_points,num_additional_samp_points)
                )
# model = Lux.Chain(SkipConnection(Lux.Chain(conv1(1, 3), conv2(3, 3), convsigm2(3, 3))
#         , connection_before_kernelA; name="prim_convs")
#         , KernelA(Nx, threads, blocks))
opt = Optimisers.Adam(0.0001)
tstate_glob = Lux.Experimental.TrainState(rng, model, opt)
tstate_glob=cu(tstate_glob)

y_pred, st = Lux.apply(model, CuArray(imagee),tstate_glob.parameters, tstate_glob.states)

out_sampled_points,tetr_dat=y_pred


sizz_out=size(out_sampled_points)
out_sampled_points_reshaped=reshape(out_sampled_points[:,:,1:2],(get_num_tetr_in_sv(),Int(round(sizz_out[1]/get_num_tetr_in_sv())),sizz_out[2],2))
size(out_sampled_points_reshaped)
out_sampled_points_reshaped=permutedims(out_sampled_points_reshaped,[2,1,3])
size(out_sampled_points_reshaped)
values=out_sampled_points_reshaped[:,:,1]
weights=out_sampled_points_reshaped[:,:,2]

size(out_sampled_points_reshaped)



consecutive_array = collect(1:48*9*2)
reshaped_array = reshape(consecutive_array, (48, 9, 2))

reshaped_array[1,1,1]
reshaped_array[25,1,1]

reshaped_array_b=reshape(reshaped_array,(24,2,9,2))
reshaped_array_b=permutedims(reshaped_array_b,[2,3,1,4])
reshaped_array_b=reshape(reshaped_array,(2,24*9,2))

reshaped_array_b[1,1,1]
reshaped_array_b[2,1,1]

reshaped_array_b[1,3,1,1]
reshaped_array_b[2,3,1,1]
a
# gs = only(gradient(p -> sum(first(Lux.apply(model, CuArray(imagee), p, st))), ps))




"""
loss function need to first take all of the sv border points and associated variance from tetr_dat
    the higher the mean of those the better 
in tetr dat for each tetrahedron first entry is sv center and the rest are border points
    we need to take the border points and get variance which is the fourth in each point data 


Next we need to take the weighted variance of the points sampled from out_sampled_points separately for each supervoxel 
    as we have all of the tetrahedrons flattened we need to reshape the outsampled points array so we will be able to process
    each supervoxel separately and calculate separately variance of each
    so we out sample points are in the shape of (num tetrahedrons, num_sample points , 5) last dimension is info about the point
    first entry is interpolated value and next one its weight

To consider - using tulio or sth with einsum to get the variance of each supervoxel

finally mean of variance of supervoxels calculated from out sampled points should be as small as possible

"""
function get_border_loss(tetr_dat)
    return mean(tetr_dat[:,2:end,4])*(-1)
end

function get_sv_variance_loss(out_sampled_points)
    num_tetr_per_sv=get_num_tetr_in_sv()
    out=out_sampled_points[:,:,1:2]
    variances=reshape(out,(num_tetr_per_sv,size(out_sampled_points)[2],2))
    # krowa check if this reshaping is doing what we want it to do
    return mean(variances)
end

function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    out_sampled_points,tetr_dat=y_pred
    b_loss=get_border_loss(tetr_dat)
    
    return b_loss, st, ()
end
# vjp = Lux.Experimental.ADTypes.AutoEnzyme()


"""
look into the synthethic data and we check weather the location of tetr dat points is etting closer to 
the original ones from synthethic data
"""
function get_metric_in_synth(tstate,tetr_out_saved,model)
    y_pred, st = Lux.apply(model, CuArray(imagee),tstate.parameters, tstate.states)
    out_sampled_points,tetr_dat=y_pred
    return mean((CuArray(tetr_out_saved[:,2:end,1:3])-tetr_dat[:,2:end,1:3]).^2)
end

function main_loop()
    tstate = Lux.Experimental.TrainState(rng, model, opt)
    tstate=cu(tstate)
    vjp = Lux.Experimental.ADTypes.AutoZygote()

    for i in 1:600
        _, loss, _, tstate = Lux.Experimental.single_train_step!(vjp, loss_function, CuArray(imagee), tstate)

            metric=get_metric_in_synth(tstate,tetr_out_saved,model)
            print("\n  *********** $(i) $(round(loss; digits = 2)) metric $(round(metric; digits = 2)) \n")

    end
end

main_loop()
# using Pkg
# Pkg.add(url="https://github.com/LuxDL/Lux.jl.git")
# gs = only(gradient(p -> sum(first(Lux.apply(model, CuArray(imagee), p, st))), ps))



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

