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

# threads_apply_gauss = (4, 4, 4)
# blocks_apply_gauss = (2, 2, 2)



struct Gauss_apply_str<: Lux.AbstractExplicitLayer
    gauss_numb::Int
    threads_apply_gauss::Tuple{Int64, Int64, Int64}
    blocks_apply_gauss::Tuple{Int64, Int64, Int64}

end

function Gauss_apply(gauss_numb::Int
    ,threads_apply_gauss::Tuple{Int64, Int64, Int64}
    ,blocks_apply_gauss::Tuple{Int64, Int64, Int64})::Gauss_apply_str
    return Gauss_apply_str(gauss_numb,threads_apply_gauss,blocks_apply_gauss)
end


"""
we will create a single variable for common stdGaus we will initialize it as a small value
    and secondly we will have the set of means for gaussians - we will set them uniformly from 0 to 1
"""
function Lux.initialparameters(rng::AbstractRNG, l::Gauss_apply_str)
    return ( (stdGaus=CuArray([Float32(0.05)]))
    ,means =  CuArray(Float32.((collect(0:(l.gauss_numb-1)))./(l.gauss_numb-1) )))
end
function Lux.initialstates(::AbstractRNG, l::Gauss_apply_str)::NamedTuple
    return (meansLength=l.gauss_numb
                        ,threads_apply_gauss=l.threads_apply_gauss
                        ,blocks_apply_gauss=l.blocks_apply_gauss)

end
# l=Gauss_apply(gauss_numb_top,threads_apply_gauss,blocks_apply_gauss)
# ps, st = Lux.setup(rng, l)


"""
voxel wise apply of the gaussian distributions
basically we want in the end to add multiple sums - hence in order to in

"""
function applyGaussKern(means,stdGaus,origArr,out,meansLength)
    #adding one becouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #iterate over all gauss parameters
    out[x,y,z]=univariate_normal(origArr[x,y,z], means[1], stdGaus[1]^2)
    for i in 2:meansLength
        #we are saving alamax of two distributions previous and current one
        out[x,y,z]= alaMax(out[x,y,z],univariate_normal(origArr[x,y,z], means[i], stdGaus[1]^2))
       # out[x,y,z]= max(univariate_normal(origArr[x,y,z], means[i-1], stdGaus[1])
        # ,univariate_normal(origArr[x,y,z], means[i], stdGaus[1]))
    end #for    
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


function (l::Gauss_apply_str)(origArr, ps, st::NamedTuple)
    return callGaussApplyKern(ps.means,ps.stdGaus,origArr
    ,st.meansLength
    ,st.threads_apply_gauss
    ,st.blocks_apply_gauss),st
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
