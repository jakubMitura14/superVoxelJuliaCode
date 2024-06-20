using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux, LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra

using Revise

includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern_unrolled.jl")


function set_tetr_dat_kern_deff(tetr_dat,d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr,d_source_arr, control_points,d_control_points, sv_centers,d_sv_centers,max_index)
    Enzyme.autodiff_deferred(Enzyme.Reverse, set_tetr_dat_kern_unrolled, Const
        , Duplicated(tetr_dat,d_tetr_dat), Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(source_arr,d_source_arr)
        , Duplicated(control_points,d_control_points), Duplicated(sv_centers,d_sv_centers),Const(max_index))
    return nothing
  end



function call_set_tetr_dat_kern(tetr_dat,source_arr, control_points, sv_centers,threads_tetr_set,blocks_tetr_set)
    tetr_dat_out=copy(tetr_dat)
    max_index=size(tetr_dat_out)[1]
    @cuda threads = threads_tetr_set blocks = blocks_tetr_set set_tetr_dat_kern_unrolled(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers,max_index)
    return tetr_dat_out
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_set_tetr_dat_kern), tetr_dat, source_arr, control_points, sv_centers,threads_tetr_set,blocks_tetr_set)

    #we get here correct tetr dat out by mutation
    tetr_dat_out=call_set_tetr_dat_kern(tetr_dat, source_arr, control_points, sv_centers,threads_tetr_set,blocks_tetr_set)


    function call_test_kernel1_pullback(d_tetr_dat_out)
       
        d_tetr_dat_out = CuArray(collect(d_tetr_dat_out))
        d_tetr_dat = CUDA.zeros(size(tetr_dat)...)
        d_source_arr = CUDA.zeros(size(source_arr)...)
        d_control_points = CUDA.zeros(size(control_points)...)
        d_sv_centers = CUDA.zeros(size(sv_centers)...)
        max_index=size(tetr_dat_out)[1]

        @cuda threads = threads_tetr_set blocks = blocks_tetr_set set_tetr_dat_kern_deff(tetr_dat,d_tetr_dat
        , tetr_dat_out, d_tetr_dat_out, source_arr,d_source_arr, control_points,d_control_points, sv_centers,d_sv_centers,max_index     )
        
        return NoTangent(), d_tetr_dat, d_source_arr, d_control_points, d_sv_centers, NoTangent(), NoTangent()
    end

    # tetr_dat_out=tetr_dat_out[1:tetr_shape[1],:,:]

    return tetr_dat_out, call_test_kernel1_pullback

end


############## lux definitions
struct Set_tetr_dat_str <: Lux.AbstractExplicitLayer
    radiuss::Float32
    pad_voxels::Int
    image_shape::Tuple{Int,Int,Int}
end

function Lux.initialparameters(rng::AbstractRNG, l::Set_tetr_dat_str)
    return ()
end
"""
check the optimal launch configuration for the kernel
calculate the number of threads and blocks and how much padding to add if needed
"""
function prepare_for_set_tetr_dat(image_shape,tetr_dat_shape)
    # ,control_points_shape,sv_centers_shape)
    # bytes_per_thread=0
    # blocks_apply_w,threads_res,maxBlocks=set_tetr_dat_kern_unrolled(Cuda.zeros(tetr_dat_shape...)
    # , Cuda.zeros(tetr_dat_shape...)
    # , Cuda.zeros(image_shape...), control_points, sv_centers,max_index)
    threads_res=256
    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads_res)

    return threads_res,needed_blocks
end
function Lux.initialstates(::AbstractRNG, l::Set_tetr_dat_str)::NamedTuple

    sv_centers,tetr_dat,dims = initialize_for_tetr_dat(l.image_shape, l.radiuss,0)

    threads_tetr_set, blocks_tetr_set = prepare_for_apply_weights_to_locs_kern(l.image_shape, size(tetr_dat))
    
    return (radiuss=l.radiuss, image_shape=l.image_shape
    , threads_tetr_set=threads_tetr_set, blocks_tetr_set=blocks_tetr_set
    , sv_centers=Float32.(sv_centers),tetr_dat=Float32.(tetr_dat),pad_voxels=l.pad_voxels)

end

function pad_source_arr(source_arr,out_arr,pad_voxels,image_shape)
    source_arr=source_arr[:,:,:,1,1]
    # p=pad_voxels*2
    p_beg=pad_voxels+1
    # new_arr=zeros(image_shape[1]+p,image_shape[2]+p,image_shape[3]+p)
    out_arr[p_beg:image_shape[1]+pad_voxels,p_beg:image_shape[2]+pad_voxels,p_beg:image_shape[3]+pad_voxels]=source_arr
    return out_arr
end





function ChainRulesCore.rrule(::typeof(pad_source_arr), source_arr,out_arr,pad_voxels,image_shape)

    #we get here correct tetr dat out by mutation
    out_arr=pad_source_arr(source_arr,out_arr,pad_voxels,image_shape)


    function call_test_kernel1_pullback(d_out_arr)
        d_out_arr = CuArray(collect(d_out_arr))
        d_source_arr = CUDA.zeros(size(source_arr)...)

        Enzyme.autodiff(Reverse, f, Duplicated(source_arr, d_source_arr), Duplicated(out_arr,d_out_arr));
        
        return NoTangent(), d_source_arr, d_out_arr, NoTangent(), NoTangent()
    end

    return out_arr, call_test_kernel1_pullback

end


function add_tetr(tetr_dat_out,out_arr,pad_voxels)

    out_arr[:,:,1:3]=(tetr_dat_out[:,:,1:3].+pad_voxels)
    out_arr[:,:,4]=tetr_dat_out[:,:,4]
    return out_arr
end    


function ChainRulesCore.rrule(::typeof(add_tetr), tetr_dat_out,out_arr,pad_voxels)

    #we get here correct tetr dat out by mutation
    out_arr=add_tetr(tetr_dat_out,out_arr,pad_voxels)


    function call_test_kernel1_pullback(d_out_arr)
        d_out_arr = CuArray(collect(d_out_arr))
        d_tetr_dat_out = CUDA.zeros(size(tetr_dat_out)...)

        Enzyme.autodiff(Reverse, f, Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(out_arr,d_out_arr));
        
        return NoTangent(), d_tetr_dat_out, d_out_arr, NoTangent()
    end

    return out_arr, call_test_kernel1_pullback

end



function (l::Set_tetr_dat_str)(x, ps, st::NamedTuple)
    control_points,source_arr = x

    tetr_dat_out=call_set_tetr_dat_kern(st.tetr_dat,source_arr, control_points, st.sv_centers,st.threads_tetr_set,st.blocks_tetr_set)
    # print("\n in lux 1  tetr_dat_out $(sum(tetr_dat_out))  \n")

    #TODO if we will deal with spacing we need to take it into account here similar if we will use batches pad_source_arr need to be modified
    tetr_dat_out=add_tetr(tetr_dat_out,CuArray(zeros(Float32,size(tetr_dat_out)...)),st.pad_voxels)
    # print("\n in lux 2  tetr_dat_out $(sum(tetr_dat_out))  \n")

    
    source_arr=pad_source_arr(source_arr,CuArray(zeros(Float32,st.image_shape[1]+(st.pad_voxels*2),st.image_shape[2]+(st.pad_voxels*2),st.image_shape[3]+(st.pad_voxels*2))),st.pad_voxels,st.image_shape)    

    return (tetr_dat_out,source_arr), st
end
