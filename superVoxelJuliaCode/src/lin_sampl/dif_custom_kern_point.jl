using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux, LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra

using Revise

includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
# Enzyme.API.strictAliasing!(false)# taken from here https://github.com/EnzymeAD/Enzyme.jl/issues/1159


# function set_point_info_kern_deff(tetr_dat,d_tetr_dat, out_sampled_points, d_out_sampled_points, source_arr,d_source_arr, num_base_samp_points,num_additional_samp_points,max_index)
function set_point_info_kern_deff(tetr, out, so, num_base_samp_points,num_additional_samp_points,max_index)

    Enzyme.autodiff_deferred(Enzyme.Reverse, point_info_kern_unrolled, Const
        , tetr
        , out
        , so
        , Const(num_base_samp_points),Const(num_additional_samp_points),Const(max_index))
    return nothing
  end




function call_point_info_kern(tetr_dat,source_arr, num_base_samp_points, num_additional_samp_points, threads_point_info,blocks_point_info)

    out_sampled_points = CUDA.zeros((size(tetr_dat)[1], num_base_samp_points + (3 * num_additional_samp_points), 5))
    max_index=size(tetr_dat)[1]
    @cuda threads = threads_point_info blocks = blocks_point_info point_info_kern_unrolled(
        tetr_dat,out_sampled_points  ,source_arr,num_base_samp_points
        ,num_additional_samp_points,max_index)

    #@device_code_warntype  
    return out_sampled_points
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_point_info_kern), tetr_dat, source_arr, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info)

    print("\n yyyyyyyyyyyyyyyyyyyyyy \n")
    out_sampled_points = call_point_info_kern(tetr_dat,source_arr, num_base_samp_points, num_additional_samp_points, threads_point_info,blocks_point_info)

    function call_test_kernel1_pullback(d_out_sampled_points)
        #@device_code_warntype 

        d_out_sampled_points = CuArray(collect(d_out_sampled_points))
        max_index=size(tetr_dat)[1]

        d_tetr_dat = CUDA.zeros(size(tetr_dat)...)
        d_source_arr = CUDA.zeros(size(source_arr)...)
        so=Duplicated(source_arr,d_source_arr)
        tetr=Duplicated(tetr_dat,d_tetr_dat)
        out=Duplicated(out_sampled_points, d_out_sampled_points)

        @cuda threads = threads_point_info blocks = blocks_point_info set_point_info_kern_deff(tetr, out, so, num_base_samp_points,num_additional_samp_points,max_index)
        # @cuda threads = threads_point_info blocks = blocks_point_info set_point_info_kern_deff(tetr_dat,d_tetr_dat, out_sampled_points
        # , d_out_sampled_points, source_arr,d_source_arr, num_base_samp_points,num_additional_samp_points,max_index)

        return NoTangent(), d_tetr_dat, d_source_arr, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return out_sampled_points, call_test_kernel1_pullback

end
############## lux definitions
struct Point_info_kern_str <: Lux.AbstractExplicitLayer
    radiuss::Float32
    image_shape::Tuple{Int,Int,Int}
    num_base_samp_points::Int
    num_additional_samp_points::Int
end



function Lux.initialparameters(rng::AbstractRNG, l::Point_info_kern_str)
    return ()
end
"""
check the optimal launch configuration for the kernel
calculate the number of threads and blocks and how much padding to add if needed
"""
function prepare_point_info_kern(image_shape,tetr_dat_shape)
    # ,control_points_shape,sv_centers_shape)
    # bytes_per_thread=0
    # blocks_apply_w,threads_res,maxBlocks=set_tetr_dat_kern_unrolled(Cuda.zeros(tetr_dat_shape...)
    # , Cuda.zeros(tetr_dat_shape...)
    # , Cuda.zeros(image_shape...), control_points, sv_centers,max_index)
    threads_res=256
    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads_res)

    return threads_res,needed_blocks
end
function Lux.initialstates(::AbstractRNG, l::Point_info_kern_str)::NamedTuple

    # num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info

    threads_point_info, blocks_point_info = prepare_point_info_kern(l.image_shape, get_tetr_dat_shape(l.radiuss,l.image_shape))
    
    return (radiuss=l.radiuss, image_shape=l.image_shape
    , threads_point_info=threads_point_info, blocks_point_info=blocks_point_info
    ,num_base_samp_points=num_base_samp_points,num_additional_samp_points=num_additional_samp_points)

end

function (l::Point_info_kern_str)(x, ps, st::NamedTuple)
    tetr_dat,source_arr = x
    out_sampled_points=call_point_info_kern(tetr_dat,source_arr, st.num_base_samp_points, st.num_additional_samp_points, st.threads_point_info,st.blocks_point_info)
    return (out_sampled_points,tetr_dat), st
end
