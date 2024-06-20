using Pkg
using ChainRulesCore,Zygote,CUDA,Enzyme
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")

"""
calculates optimal number of threads and blocks for given kernel
kernel_fun - kernel function for which we want to calculate the number of threads and blocks
args - arguments that are used by the kernel 
bytes_per_thread- indicate how much shared memory is used by single thread
"""
function computeBlocksFromOccupancy(kernel_fun,args, bytes_per_thread)

    wanted_threads =10000000
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            true ? prevwarp(device(), max_threads) : max_threads
        else
            wanted_threads
        end
    end
    compute_shmem(threads) = Int64(threads * bytes_per_thread )
    
       kernel = @cuda launch=false kernel_fun(args...) 
       kernel_config = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
       blocks =  kernel_config.blocks
       threads =  kernel_config.threads
       maxBlocks = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    
return blocks,threads,maxBlocks
end

"""
check the optimal launch configuration for the kernel
calculate the number of threads and blocks and how much padding to add if needed
"""
function prepare_for_apply_weights_to_locs_kern(control_points_shape,weights_shape)
    bytes_per_thread=0
    blocks_apply_w,threads_apply_w,maxBlocks=computeBlocksFromOccupancy(apply_weights_to_locs_kern,(CUDA.ones(control_points_shape...),CUDA.ones(control_points_shape...),CUDA.ones(weights_shape...)
    ,Float32(3),UInt32(control_points_shape[1]),UInt32(control_points_shape[2]),UInt32(control_points_shape[3])
    ,UInt32(0),UInt32(0),UInt32(0)
    ), bytes_per_thread)
    # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
    # needed_blocks=ceil(total_num / threads_apply_w)
    # threads_res=(8,8,(floor(Int,threads_apply_w/64)))
    threads_res=(8,8,(floor(Int,threads_apply_w/128)))
    needed_blocks=(ceil(Int,control_points_shape[1]/threads_res[1]),ceil(Int,control_points_shape[2]/threads_res[2]),ceil(Int,control_points_shape[3]/threads_res[3]))

    return threads_res,needed_blocks
end