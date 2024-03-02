using Pkg
using ChainRulesCore,Zygote,CUDA,Enzyme

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