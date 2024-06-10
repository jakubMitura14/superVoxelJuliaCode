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

includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern_unrolled.jl")


function set_tetr_dat_kern_deff(tetr_dat,d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr,d_source_arr, control_points,d_control_points, sv_centers,d_sv_centers)
    Enzyme.autodiff_deferred(Enzyme.Reverse, set_tetr_dat_kern_unrolled, Const
        , Duplicated(tetr_dat,d_tetr_dat), Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(source_arr,d_source_arr)
        , Duplicated(control_points,d_control_points), Duplicated(sv_centers,d_sv_centers))
    return nothing
end

"""
pad tetr data with constant in order to avoid padding in the kernel
we use 2 as default not 0 in order to avoid it trying to get things at index 0 what in julia is impossible
"""
function pad_tetr(arr,pad_point_info,tetr_shape,constt=2)
    to_pad_tetr = CUDA.ones(pad_point_info+tetr_shape[1] , tetr_shape[2], tetr_shape[3])*constt
    to_pad_tetr[(pad_point_info+1):end , :, :]=arr
    return to_pad_tetr
end



function call_set_tetr_dat_kern(tetr_dat,tetr_dat_out, source_arr, control_points, sv_centers)

    # tetr_shape = size(tetr_dat)

    # tetr_dat = pad_tetr(tetr_dat,pad_point_info,tetr_shape)
    # tetr_dat_out = CUDA.zeros(size(tetr_dat)...)

    # @cuda threads = threads blocks = blocks set_tetr_dat_kern_unrolled(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers)

    # tetr_dat_out = tetr_dat_out[1:tetr_shape[1], :, :]
    # tetr_dat_out = CUDA.zeros(size(tetr_dat)...)
    dev = get_backend(tetr_dat)
    set_tetr_dat_kern_unrolled(dev, 256)(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, ndrange=(size(tetr_dat)[1]))
    KernelAbstractions.synchronize(dev)
    return nothing
    # return tetr_dat_out
end




# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_set_tetr_dat_kern), tetr_dat,tetr_dat_out, source_arr, control_points, sv_centers)

    #we get here correct tetr dat out by mutation
    call_set_tetr_dat_kern(tetr_dat, tetr_dat_out,source_arr, control_points, sv_centers)



    function call_test_kernel1_pullback(d_tetr_dat_out)
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)

        # d_tetr_dat_out = vcat(CuArray(collect(d_tetr_dat_out)), to_pad_tetr)
        # tetr_dat = pad_tetr(tetr_dat,pad_point_info,tetr_shape)          
        d_tetr_dat_out = CuArray(collect(d_tetr_dat_out))
        # d_tetr_dat_out = pad_tetr(d_tetr_dat_out,pad_point_info,tetr_shape,1)
        d_tetr_dat = CUDA.zeros(size(tetr_dat)...)

        d_source_arr = CUDA.zeros(size(source_arr)...)
        d_control_points = CUDA.zeros(size(control_points)...)
        d_sv_centers = CUDA.zeros(size(sv_centers)...)  
        
        Duplicated(tetr_dat,d_tetr_dat)
        Duplicated(tetr_dat_out, d_tetr_dat_out)
        Duplicated(source_arr,d_source_arr)
        Duplicated(control_points,d_control_points)
        Duplicated(sv_centers,d_sv_centers)


        Enzyme.autodiff_deferred(Enzyme.Reverse, call_set_tetr_dat_kern, Const
        , Duplicated(tetr_dat,d_tetr_dat), Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(source_arr,d_source_arr)
        , Duplicated(control_points,d_control_points), Duplicated(sv_centers,d_sv_centers))
        # Enzyme.autodiff(Enzyme.Reverse, set_tetr_dat_kern_unrolled, Const
        # , Duplicated(tetr_dat,d_tetr_dat), Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(source_arr,d_source_arr)
        # , Duplicated(control_points,d_control_points), Duplicated(sv_centers,d_sv_centers))

        # # threads_point_info=128#TODO remove
        # # blocks_point_info=1#TODO remove
        # #TODO consider add padding to the source array
        # @cuda threads = threads_point_info blocks = blocks_point_info set_tetr_dat_kern_deff(tetr_dat,d_tetr_dat
        # , tetr_dat_out, d_tetr_dat_out, source_arr,d_source_arr, control_points,d_control_points, sv_centers,d_sv_centers     )
        # #reverse padding
        # d_tetr_dat = d_tetr_dat[1:tetr_shape[1], :, :]

        return NoTangent(), d_tetr_dat,d_source_arr, d_control_points,d_sv_centers, NoTangent(), NoTangent(), NoTangent()
    end

    # tetr_dat_out=tetr_dat_out[1:tetr_shape[1],:,:]

    return tetr_dat_out, call_test_kernel1_pullback

end
