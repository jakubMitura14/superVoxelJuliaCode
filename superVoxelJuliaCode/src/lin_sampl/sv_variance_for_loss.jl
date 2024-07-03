using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux, LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra

using Revise
using LLVMLoopInfo

"""
after getting data of out sampled points it will perform the calculation of weighted variance for each super voxel 

IMPORTANT kernel not fully optimized - the size of the shared memory is not optimal; the size of thread blocks is not divisible by 32
    we do not use between thread communication just communication via shared memory; also I use sync threads instead of more granular synchonizations
    """
function get_per_sv_variance(out_sampled_points,out_res)

    shared_arr = CuStaticSharedArray(Float32, (24,9,2))
    shared_arr_b = CuStaticSharedArray(Float32, (2))

    index = (1 + ((blockIdx().x - 1) * CUDA.blockDim_x()))

    #loading interpolated values
    shared_arr[threadIdx().y,threadIdx().z,1]=out_sampled_points[index,threadIdx().y,threadIdx().z,1]
    #loading weights
    shared_arr[threadIdx().y,threadIdx().z,2]=out_sampled_points[index,threadIdx().y,threadIdx().z,2]
    #sync threads
    sync_threads()
    #calculating sum of values and weights on sv dimension 
    if(threadIdx().z==1)
        shared_arr[threadIdx().y,1,1]=(shared_arr[threadIdx().y,1,1]*shared_arr[threadIdx().y,1,2])
        for i in 2:CUDA.blockDim_z()
            shared_arr[threadIdx().y,1,1]+=(shared_arr[threadIdx().y,i,1]*shared_arr[threadIdx().y,i,2])
            shared_arr[threadIdx().y,1,2]+=shared_arr[threadIdx().y,i,2]
        end
    end    

    sync_threads()
    #reducing sum of values and weights on points dimension
    if(threadIdx().z==1 && threadIdx().y==1)
        for i in 2:CUDA.blockDim_y()
            shared_arr[1,1,1]+=shared_arr[i,1,1]
            shared_arr[1,1,2]+=shared_arr[i,1,2]
        end
        #now we have all summed up on first entry in shared memory we can save it to as mean
        shared_arr_b[1]=(shared_arr[1,1,1]/shared_arr[1,1,2])
    end    
    sync_threads()

    out_res[index]=shared_arr_b[1]#TODO remove
    # out_res[index]=shared_arr_b[1]#TODO remove

    ############# caclulating variance

    # #loading interpolated values and subtracting mean of sv
    # shared_arr[threadIdx().y,threadIdx().z,1]=((out_sampled_points[index,threadIdx().y,threadIdx().z,1]-shared_arr_b[1])^2)
    # #loading weights one more time
    # shared_arr[threadIdx().y,threadIdx().z,2]=out_sampled_points[index,threadIdx().y,threadIdx().z,2]
    # #sync threads
    # sync_threads()
    # #calculating sum of values and weights on sv dimension 
    # if(threadIdx().z==1)
    #     shared_arr[threadIdx().y,1,1]=(shared_arr[threadIdx().y,1,1]*shared_arr[threadIdx().y,1,2])
    #     for i in 2:CUDA.blockDim_z()
    #         shared_arr[threadIdx().y,1,1]+=(shared_arr[threadIdx().y,i,1]*shared_arr[threadIdx().y,i,2])
    #     end
    # end    
    # sync_threads()
    # if(threadIdx().z==2)
    #     for i in 2:CUDA.blockDim_z()
    #         shared_arr[threadIdx().y,1,2]+=shared_arr[threadIdx().y,i,2]
    #     end
    # end    
    # sync_threads()
    # #reducing sum of values and weights on points dimension
    # if(threadIdx().z==1 && threadIdx().y==1)
    #     for i in 2:CUDA.blockDim_y()
    #         shared_arr[1,1,1]+=shared_arr[i,1,1]
    #         shared_arr[1,1,1]+=shared_arr[i,1,2]
    #     end
    #     #now we have all summed up on first entry in shared memory we can save variance of global memory
    #     out_res[index]=(shared_arr[1,1,1]/shared_arr[1,1,2])
    # end    


    return nothing
end


function get_per_sv_variance_deff(out_sampled_points,d_out_sampled_points,out_res,d_out_res)

    Enzyme.autodiff_deferred(Enzyme.Reverse, get_per_sv_variance, Const
        ,Duplicated(out_sampled_points,d_out_sampled_points)
        ,Duplicated(out_res,d_out_res)
        )   
         return nothing
  end




function call_get_per_sv_variance(out_sampled_points)
    
    blocks_x=size(out_sampled_points)[1]
    out_res=CUDA.zeros(blocks_x)
    @cuda threads = (1,24,9) blocks = (blocks_x,1,1) get_per_sv_variance(out_sampled_points,out_res)

    return out_res
end


function ChainRulesCore.rrule(::typeof(call_get_per_sv_variance), out_sampled_points)

    out_res = call_get_per_sv_variance(out_sampled_points)

    function call_test_kernel1_pullback(d_out_res)
        #@device_code_warntype 

        d_out_res = CuArray(collect(d_out_res))
        d_out_sampled_points = CUDA.zeros(size(out_sampled_points)...)
        blocks_x=size(out_sampled_points)[1]

        @cuda threads = (1,24,9) blocks = (blocks_x,1,1) get_per_sv_variance_deff(out_sampled_points,d_out_sampled_points,out_res,d_out_res)


        return NoTangent(), d_out_sampled_points
    end

    return out_res, call_test_kernel1_pullback

end



#block dim x should be 1 ,block dim y should be 24 and z should be 9