"""
define the lux layer with custom backpropagation through Enzyme
will give necessery data for loss function and for graph creation from 
    supervoxels
"""

using ChainRulesCore,Zygote,CUDA,Enzyme
using CUDAKernels
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays

rng = Random.default_rng()

Nx, Ny, Nz = 8, 8, 8
oneSidePad = 1
totalPad=oneSidePad*2
A = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dA= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

Aoutout = CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dAoutout= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

p = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dp= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

#how many gaussians we will specify 
const gauss_numb_top = 8

threads_apply_gauss = (4, 4, 4)
blocks_apply_gauss = (2, 2, 2)

struct Gauss_apply_str<: Lux.AbstractExplicitLayer
    gauss_numb::Int
end

function Gauss_apply(gauss_numb::Int)
    return Gauss_apply_str(gauss_numb)
end


"""
we will create a single variable for common variance we will initialize it as a small value
    and secondly we will have the set of means for gaussians - we will set them uniformly from 0 to 1
"""
function Lux.initialparameters(rng::AbstractRNG, l::Gauss_apply_str)
    return ((variance=0.05)
    ,means =  CuArray(Float32.((collect(0:(l.gauss_numb-1)))./(l.gauss_numb-1) )))
end
Lux.initialstates(::AbstractRNG, ::Gauss_apply_str) = NamedTuple()

l=Gauss_apply(gauss_numb_top)
ps, st = Lux.setup(rng, l)


"""
voxel wise apply of the gaussian distributions
"""
function applyGaussKern(means,variance,indArr,out,meansLength)
    #adding one becouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #iterate over all gauss parameters
    for i in 2:meansLength
        #we are saving alamax of two distributions previous and current one
        out= alaMax(univariate_normal(indArr[x,y,z], means[i-1], variance)
        ,univariate_normal(indArr[x,y,z], means[i], variance))
    end #for    
    return nothing
end
"""
Enzyme definitions for calculating derivatives of applyGaussKern in back propagation
"""
function applyGaussKern_Deff(means,d_means,variance,indArr,d_indArr,out,d_out,meansLength)
    Enzyme.autodiff_deferred(applyGaussKern, Const
    ,Duplicated(means, d_means)
    ,Active(variance)
    ,Duplicated(indArr, d_indArr)
    ,Duplicated(out, d_out)
    ,Const(meansLength)    )
    return nothing
end

"""
call function with out variable initialization
"""
function callGaussApplyKern(means,variance,indArr,meansLength)
    out = CUDA.zeros(size(indArr)) 
    @cuda threads = threads_apply_gauss blocks = blocks_apply_gauss testKern( A, p,  Aout)
    return Aout
end

# aa=calltestKern(A, p)
# maximum(aa)

# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(callGaussApplyKern), means,variance,indArr,meansLength)
    out = calltestKern(A, p)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)
        # Allocate shadow memory.
        threads = (4, 4, 4)
        blocks = (2, 2, 2)
        dp = CUDA.ones(size(p))
        dA = CUDA.ones(size(A))
        @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)))

        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp
        
        return f̄, x̄, ȳ
    end   
    return Aout, call_test_kernel1_pullback

end
#first testing
# ress=Zygote.jacobian(calltestKern,A, p,Nx )



typeof(ress)
maximum(ress[1])
maximum(ress[2])



# # But still recommened to define these
# Lux.parameterlength(l::KernelAstr) = l.out_dims * l.in_dims + l.out_dims

# Lux.statelength(::KernelAstr) = 0

function (l::KernelAstr)(x, ps, st::NamedTuple)
    return calltestKern(x, ps.paramsA),st
end

rng = Random.default_rng()
oneSidePad = 1
totalPad=oneSidePad*2  
Nx, Ny, Nz = 8+totalPad, 8+totalPad, 8+totalPad

l = KernelA(Nx)

ps, st = Lux.setup(rng, l)

println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
        Lux.statelength(l))


x = randn(rng, Float32, Nx, Ny,Nz)
x= CuArray(x)

y_pred, st =Lux.apply(l, x, ps, st) # or `l(x, ps, st)`

model = Lux.Chain(KernelA(Nx),KernelA(Nx) )
opt = Optimisers.Adam(0.0003)

"""
extremely simple loss function - that just wan to decrese sumof all inputs
"""
function loss_function(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    return (100-sum(y_pred))^2, st, ()
end

tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
#tstate = Lux.Training.TrainState(rng, model, opt)
vjp_rule = Lux.Training.ZygoteVJP()


function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data,
    epochs::Int)
   # data = data .|> Lux.gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
                                                                data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

tstate = main(tstate, vjp_rule, x,1)


tstate = main(tstate, vjp_rule, x,1000)

# using ChainRulesTestUtils
# test_rrule(testKern,A, p, Aout)

# a=Fill(7.0f0, 3, 2)
# collect(a)
