using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux, LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays
using LinearAlgebra
using Images, ImageFiltering
using Revise

includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")


function set_tetr_dat_kern_deff(tetr_dat, d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, num_base_samp_points, num_additional_samp_points
)




    # shared_arr = CuStaticSharedArray(Float32, (128,3))
    # d_shared_arr = CuStaticSharedArray(Float32, (128,3))

    Enzyme.autodiff_deferred(Enzyme.Reverse, set_tetr_dat_kern, Const
        # , Duplicated(CuStaticSharedArray(Float32, (128,3)), CuStaticSharedArray(Float32, (128,3)))
        , Duplicated(tetr_dat, d_tetr_dat), Duplicated(tetr_dat_out, d_tetr_dat_out), Const(source_arr)
        # , Duplicated(control_points, d_control_points)
        , Const(control_points), Duplicated(sv_centers, d_sv_centers), Const(num_base_samp_points), Const(num_additional_samp_points))
    return nothing
end


function call_set_tetr_dat_kern(tetr_dat, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads, blocks, pad_point_info)
    # shared_arr = CuStaticSharedArray(Float32, (threads[1],3))
    # shared_arr = CuStaticSharedArray(Float32, (100,3))
    # shared_arr = CuDynamicSharedArray(Float32, (threads[1],3))
    #shmem is in bytes
    tetr_shape = size(tetr_dat)
    to_pad_tetr = CUDA.ones(pad_point_info, tetr_shape[2], tetr_shape[3]) * 2
    tetr_dat = vcat(tetrs, to_pad_tetr)
    tetr_dat_out = CUDA.zeros(size(tetr_dat)...)


    # @cuda threads = threads blocks = blocks point_info_kern(CuStaticSharedArray(Float32, (128,3)),tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points)
    @cuda threads = threads blocks = blocks set_tetr_dat_kern(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points)


    tetr_dat_out = tetr_dat_out[1:tetr_shape[1], :, :]

    # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
    return tetr_dat_out
end




# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_set_tetr_dat_kern), tetr_dat, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, pad_point_info)


    tetr_dat_out = call_set_tetr_dat_kern(tetr_dat, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, pad_point_info)    #TODO unhash
    d_tetr_dat = CUDA.ones(size(tetr_dat)...)
    d_source_arr = CUDA.ones(size(source_arr)...)
    d_control_points = CUDA.ones(size(control_points)...)
    d_sv_centers = CUDA.ones(size(sv_centers)...)


    #pad to avoid conditionals in kernel
    tetr_shape = size(tetr_dat)
    to_pad_tetr = CUDA.ones(pad_point_info, tetr_shape[2], tetr_shape[3])
    tetr_dat = vcat(tetrs, to_pad_tetr)
    tetr_dat_out = vcat(tetr_dat_out, to_pad_tetr)
    d_tetr_dat = vcat(d_tetr_dat, to_pad_tetr)



    function call_test_kernel1_pullback(d_tetr_dat_out)
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)

        d_tetr_dat_out = vcat(CuArray(collect(d_tetr_dat_out)), to_pad_tetr)

        # d_out_sampled_points = CUDA.ones(size(out_sampled_points)...) #TODO(remove)
        # d_out_sampled_points=vcat(d_out_sampled_points,to_pad_out)#TODO(remove)
        #@device_code_warntype 
        # @device_code_warntype  @cuda threads = threads_point_info blocks = blocks_point_info Enzyme.autodiff_deferred(Enzyme.Reverse,point_info_kern, Const
        #                                                                         , Duplicated(tetr_dat, d_tetr_dat)
        #                                                                         , Duplicated(out_sampled_points, d_out_sampled_points)
        #                                                                         , Duplicated(source_arr, d_source_arr)
        #                                                                         , Duplicated(control_points, d_control_points)
        #                                                                         , Duplicated(sv_centers, d_sv_centers)
        #                                                                         ,Const(num_base_samp_points)
        #                                                                         ,Const(num_additional_samp_points) 
        # )  

        # threads_point_info=32 #TODO remove
        @device_code_warntype @cuda threads = threads_point_info blocks = blocks_point_info set_tetr_dat_kern_deff(tetr_dat, d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, num_base_samp_points, num_additional_samp_points
        )
        #reverse padding



        d_tetr_dat = d_tetr_dat[1:tetr_shape[1], :, :]

        return NoTangent(), d_tetr_dat, d_source_arr, d_control_points, d_sv_centers, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    # tetr_dat=tetr_dat[1:tetr_shape[1],:,:]


    return tetr_dat_out, call_test_kernel1_pullback

end
