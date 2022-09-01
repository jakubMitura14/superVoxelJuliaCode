#based on https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
# and https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
#so basically convolution is matrix multiplication of kernel and chunk of the array around all pixels that is the same size as the kernel
#about channels ...
# """ 
# Checking if there are mutliple channels for the single filter.
# If so, then each channel will convolve the image.
# The result of all convolutions are summed to return a single feature map.
# """

# """
# define the lux layer with custom backpropagation through Enzyme
# will give necessery data for loss function and for graph creation from 
#     supervoxels
# """



using Revise
# includet("/media/jakub/NewVolume/projects/superVoxelJuliaCode/utils/includeAll.jl")
using ChainRulesCore,Zygote,CUDA,Enzyme
using CUDAKernels
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays

# rng = Random.default_rng()

# Nx, Ny, Nz = 8, 8, 8
# oneSidePad = 1
# totalPad=oneSidePad*2
# crossBorderWhere=4
# origArr,indArr =createTestDataFor_Clustering(Nx, Ny, Nz, oneSidePad, crossBorderWhere)




#how many gaussians we will specify 
# const gauss_numb_top = 8

# threads_apply_cconv = (4, 4, 4)
# blocks_apply_cconv = (2, 2, 2)



struct Cconv_str<: Lux.AbstractExplicitLayer
    threads_apply_cconv::Tuple{Int64, Int64, Int64}
    blocks_apply_cconv::Tuple{Int64, Int64, Int64}

end

function Cconv(threads_apply_cconv::Tuple{Int64, Int64, Int64}
    ,blocks_apply_cconv::Tuple{Int64, Int64, Int64})::Cconv_str
    return Cconv_str(threads_apply_gauss,blocks_apply_gauss)
end


"""
we will create a single variable for common stdGaus we will initialize it as a small value
    and secondly we will have the set of means for gaussians - we will set them uniformly from 0 to 1
"""
function Lux.initialparameters(rng::AbstractRNG, l::Cconv_str)
    return ( (stdGaus=CuArray([Float32(0.05)]))
    ,means =  CuArray(Float32.((collect(0:(l.gauss_numb-1)))./(l.gauss_numb-1) )))
end
function Lux.initialstates(::AbstractRNG, l::Cconv_str)::NamedTuple
    return (meansLength= threads_apply_gauss=l.threads_apply_gauss
                        ,blocks_apply_gauss=l.blocks_apply_gauss)

end
# l=Gauss_apply(gauss_numb_top,threads_apply_gauss,blocks_apply_gauss)
# ps, st = Lux.setup(rng, l)


"""

"""
function applyGaussKern(means,stdGaus,origArr,out,meansLength)
    #adding one becouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #idea is to get some dose of rotation invariance - so algorithm should look a bit like a lidar - so 
    #with the cone like area in two directions at once - so algorithm in principle should compare the areas 
    # and spit out the    
 
    return nothing
end


"""



Enzyme definitions for calculating derivatives of applyGaussKern in back propagation
"""
function applyGaussKern_Deff(means,d_means,stdGaus,d_stdGaus,origArr
    ,d_origArr,out,d_out,meansLength)
    
    Enzyme.autodiff_deferred(applyGaussKern, Const
    ,Duplicated(means, d_means)
    ,Duplicated(stdGaus,d_stdGaus)
    ,Duplicated(origArr, d_origArr)
    ,Duplicated(out, d_out)
    ,Const(meansLength)    )
    return nothing
end

"""
call function with out variable initialization
"""
function callGaussApplyKern(means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss)
    out = CUDA.zeros(size(origArr)) 
    @cuda threads = threads_apply_gauss blocks = blocks_apply_gauss applyGaussKern(means,stdGaus,origArr,out,meansLength)
    return out
end

# aa=calltestKern(A, p)
# maximum(aa)

# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(callGaussApplyKern),means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss)
    out = callGaussApplyKern(means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss)
    function call_test_kernel1_pullback(d_out_prim)
        # Allocate shadow memory.
        d_means = CUDA.ones(size(means))
        d_stdGaus = CUDA.ones(size(stdGaus))
        d_origArr = CUDA.ones(size(origArr))
        d_out = CuArray(collect(d_out_prim))
        @cuda threads = threads_apply_gauss blocks = blocks_apply_gauss applyGaussKern_Deff(means,d_means,stdGaus,d_stdGaus,origArr,d_origArr,out,d_out,meansLength)
        f̄ = NoTangent()

        return f̄, d_means,d_stdGaus, d_origArr, NoTangent(), NoTangent(), NoTangent()
    end   
    return out, call_test_kernel1_pullback

end


function (l::Cconv_str)(origArr, ps, st::NamedTuple)
    return callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss),st
end


we can test on our 8 block test cube 
    and as the loss function get the 
    sum of the probabilities in non edge areas minus sum of probabilities in real edge areas