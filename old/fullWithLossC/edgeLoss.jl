"""
we will define here just the loss function that will detect basically edges
    or edge like structures
    so we will look for the places thatin some axis has high variance of the features analyzed
    so it would suggest that this is a spot where those feature changes from one area to the other ...
    
"""

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

# threads_apply_gauss = (4, 4, 4)
# blocks_apply_gauss = (2, 2, 2)



struct Gauss_apply_str<: Lux.AbstractExplicitLayer
    gauss_numb::Int
    threads_apply_gauss::Tuple{Int64, Int64, Int64}
    blocks_apply_gauss::Tuple{Int64, Int64, Int64}
    currGaussIndex::Int

end

function Gauss_apply(gauss_numb::Int
    ,threads_apply_gauss::Tuple{Int64, Int64, Int64}
    ,blocks_apply_gauss::Tuple{Int64, Int64, Int64},currGaussIndex::Int)::Gauss_apply_str
    return Gauss_apply_str(gauss_numb,threads_apply_gauss,blocks_apply_gauss,currGaussIndex)
end


"""
we will create a single variable for common stdGaus we will initialize it as a small value
    and secondly we will have the set of means for gaussians - we will set them uniformly from 0 to 1
"""
function Lux.initialparameters(rng::AbstractRNG, l::Gauss_apply_str)
    return ( (stdGaus=CUDA.ones(l.gauss_numb ))
    ,means =  CuArray(Float32.((collect(0:(l.gauss_numb-1)))./(l.gauss_numb-1) )))
end
function Lux.initialstates(::AbstractRNG, l::Gauss_apply_str)::NamedTuple
    return (meansLength=l.gauss_numb
                        ,threads_apply_gauss=l.threads_apply_gauss
                        ,blocks_apply_gauss=l.blocks_apply_gauss
                        ,currGaussIndex=l.currGaussIndex
                        
                        )

end
# l=Gauss_apply(gauss_numb_top,threads_apply_gauss,blocks_apply_gauss)
# ps, st = Lux.setup(rng, l)




"""
evaluate gaussians at 2 points, does soft thresholding and multiplies them
then their product is used as a scaling factor 
"""
@inline function multThreshold(a,b,mean,std,fa1,fa2,fb1,fb2)
    return (softThreshold_half(univariate_normal(a, mean, std^2)+univariate_normal(b, mean, std^2)))*
    (((fa1-fa2)^2) + ((fb1-fb2)^2))
end#multiply_and_threshold

"""
iterates around and executes multThreshold
"""
@inline function thresholdIter(arr,mean,std,f1Channel,f2channel,x,y,z,xChange,yChange,zChange)
    return multThreshold(arr[x,y,z,1,1],arr[x+xChange,y+yChange,z+zChange,1,1],mean,std,arr[x,y,z,f1Channel,1]
        ,arr[x+xChange,y+yChange,z+zChange,f1Channel,1],arr[x,y,z,f2channel,1],arr[x+xChange,y+yChange,z+zChange,f2channel,1])
end    

@inline function thresholdIterAround(arr,mean,std,f1Channel,f2channel,x,y,z)
    return thresholdIter(arr,mean,std,f1Channel,f2channel,x,y,z,-1,0,0)
            +thresholdIter(arr,mean,std,f1Channel,f2channel,x,y,z,1,0,0)
            +thresholdIter(arr,mean,std,f1Channel,f2channel,x,y,z,0,1,0)
            +thresholdIter(arr,mean,std,f1Channel,f2channel,x,y,z,0,-1,0)
            +thresholdIter(arr,mean,std,f1Channel,f2channel,x,y,z,0,0,1)
            +thresholdIter(arr,mean,std,f1Channel,f2channel,x,y,z,0,0,-1)
end 


"""
voxel wise apply of the gaussian distributions
basically we want in the end to add multiple sums - hence in order to in
"""
function applyGaussKern(means,stdGaus,origArr,out,meansLength,currGaussIndex)
    
    #adding one becouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #iterate over all gauss parameters and check weather some evaluated to high value
    out[x,y,z]-=(univariate_normal(origArr[x,y,z,1,1], means[currGaussIndex], stdGaus[1]^2))/100
    out[x,y,z]+= thresholdIterAround(origArr,means[currGaussIndex],stdGaus[1],3,4,x,y,z)
    # out[x,y,z]-=(univariate_normal(origArr[x,y,z,1,1], means[currGaussIndex], stdGaus[currGaussIndex]^2))/100
    # out[x,y,z]+= thresholdIterAround(origArr,means[currGaussIndex],stdGaus[currGaussIndex],3,4,x,y,z)
    # for i in 1:meansLength
    # for i in 2:3
    #     currMax = alaMax(currMax,univariate_normal(origArr[x,y,z,1,1], means[i], stdGaus[i]^2))
    #     #out[x,y,z]+= thresholdIterAround(origArr,means[i],stdGaus[i],3,4,x,y,z)
    # end #for
    # #we want to have large values evrywhere
    # out[x,y,z]-= currMax


    return nothing
end


# function applyGaussKern(means,stdGaus,origArr,out,meansLength)
#     #adding one becouse of padding
#     x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
#     y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
#     z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
#     #iterate over all gauss parameters
#     out[x,y,z]=univariate_normal(origArr[x,y,z], means[1], stdGaus[1]^2)
#     for i in 2:meansLength
#         #we are saving alamax of two distributions previous and current one
#         #out[x,y,z]= alaMax(out[x,y,z],univariate_normal(origArr[x,y,z], means[i], stdGaus[i]^2))
#         out[x,y,z]= max(out[x,y,z],univariate_normal(origArr[x,y,z], means[i], stdGaus[i]^2))
#        # out[x,y,z]= max(univariate_normal(origArr[x,y,z], means[i-1], stdGaus[1])
#         # ,univariate_normal(origArr[x,y,z], means[i], stdGaus[1]))
#     end #for    
#     return nothing
# end



"""



Enzyme definitions for calculating derivatives of applyGaussKern in back propagation
"""
function applyGaussKern_Deff(means,d_means,stdGaus,d_stdGaus,origArr
    ,d_origArr,out,d_out,meansLength,currGaussIndex)
    
    Enzyme.autodiff_deferred(applyGaussKern, Const
    ,Duplicated(means, d_means)
    ,Duplicated(stdGaus,d_stdGaus)
    ,Duplicated(origArr, d_origArr)
    ,Duplicated(out, d_out)
    ,Const(meansLength)  
    ,Const(currGaussIndex)      
    )
    return nothing
end

"""
call function with out variable initialization
"""
function callGaussApplyKern(means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss,currGaussIndex)
    out = CUDA.zeros(size(origArr)[1],size(origArr)[2],size(origArr)[3] ) 
    @cuda threads = threads_apply_gauss blocks = blocks_apply_gauss applyGaussKern(means,stdGaus,origArr,out,meansLength,currGaussIndex)
    return out
end

# aa=calltestKern(A, p)
# maximum(aa)

# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(callGaussApplyKern),means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss,currGaussIndex)
    out = callGaussApplyKern(means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss,currGaussIndex)
    function call_test_kernel1_pullback(d_out_prim)
        # Allocate shadow memory.
        d_means = CUDA.ones(size(means))
        d_stdGaus = CUDA.ones(size(stdGaus))
        d_origArr = CUDA.ones(size(origArr))
        d_out = CuArray(collect(d_out_prim))
        @cuda threads = threads_apply_gauss blocks = blocks_apply_gauss applyGaussKern_Deff(means,d_means,stdGaus,d_stdGaus,origArr,d_origArr,out,d_out,meansLength,currGaussIndex)
        f̄ = NoTangent()
        return f̄, d_means,d_stdGaus, d_origArr, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end   
    return out, call_test_kernel1_pullback

end

function (l::Gauss_apply_str)(origArr, ps, st::NamedTuple)
    
    imageView=view(origArr,:,:,:,1,:)
    imageView.-minimum(imageView)
    imageView./maximum(imageView)

    out1=callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss
    ,1  )

    out2=callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss
    ,2 )
    out3=callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss
    ,3)
    out4=callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss
    ,4 )
    out5=callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss
    ,5 )
    out6=callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss
    ,6)
    out7=callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss
    ,7)
    out8=callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss
    ,8)


    return (out1+out3+out4+out5+out6+out7+out8),st
    # return callGaussApplyKern(ps.means,ps.stdGaus,origArr
    # ,st.meansLength
    # ,st.threads_apply_gauss
    # ,st.blocks_apply_gauss
    # ,st.currGaussIndex
    # ),st
end

# l = Gauss_apply(gauss_numb_top,threads_apply_gauss,blocks_apply_gauss)
# ps, st = Lux.setup(rng, l)

# println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
#         Lux.statelength(l))

# y_pred, st =Lux.apply(l, CuArray(origArr), ps, st)

# opt = Optimisers.NAdam()
#opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0), Optimisers.NAdam());

# """
# extremely simple loss function - that just wan to decrese sumof all inputs
# """
# function loss_function(model, ps, st, x)
#     y_pred, st = Lux.apply(model, x, ps, st)
#     return -1*(sum(y_pred)), st, ()
# end

# tstate = Lux.Training.TrainState(rng, l, opt; transform_variables=Lux.gpu)
# #tstate = Lux.Training.TrainState(rng, model, opt)
# vjp_rule = Lux.Training.ZygoteVJP()


# function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data,
#     epochs::Int)
#    # data = data .|> Lux.gpu
#     for epoch in 1:epochs
#         grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
#                                                                 data, tstate)
#         @info epoch=epoch loss=loss
#         tstate = Lux.Training.apply_gradients(tstate, grads)
#     end
#     return tstate
# end

# tstate = main(tstate, vjp_rule, CuArray(origArr),1)


# tstate = main(tstate, vjp_rule,  CuArray(origArr),1000)

# using ChainRulesTestUtils
# test_rrule(testKern,A, p, Aout)

# a=Fill(7.0f0, 3, 2)
# collect(a)
