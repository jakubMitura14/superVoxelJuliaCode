using ChainRulesCore
using CUDA
using CUDAKernels
using Enzyme
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays

# a=Fill(7.0f0, 3, 2)
# collect(a)


nSize = 3


function mul_kernel(A,p,Aout)
    #adding one bewcouse of padding
    x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))+1
    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))+1
    z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))+1
    Aout[x,y,z]= p[x,y,z]*p[x,y,z]
 
    return nothing
end

function grad_mul_kernel(A, dA
    ,p,dp,Aout#::CuArray{Float32, 3}
    ,dAout)
    Enzyme.autodiff_deferred(mul_kernel, Const
    , Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout,dAout)
    )
    return nothing
end



# Function calls to allow easier high-level code.
function call_example_kernel1(A,p)
    Aout=CUDA.ones(Float32,size(x))
    @cuda threads= (8,8,8) blocks=(8,8,8) mul_kernel(A, p,Aout)
    return Aout
    #return out
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_example_kernel1), A, p)
    Aout=CUDA.ones(Float32,size(A))
    #call_example_kernel1(A, p,Aout)

    function call_example_kernel1_pullback(dAout)
        # Allocate shadow memory.
        dp = CUDA.ones(size(p))
        dA = CUDA.ones(size(A))

        # @cuda threads= (8,8,8) blocks=(8,8,8) grad_mul_kernel(A, dA,p,dp,collect(Aout),dAout)
        @cuda threads= (8,8,8) blocks=(8,8,8) grad_mul_kernel(A, dA,p,dp,collect(Aout),dAout)


        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp
        
        return f̄, x̄, ȳ
    end
    
    return Aout, call_example_kernel1_pullback
end




#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct KernelAstr<: Lux.AbstractExplicitLayer
    confA::Int
end

function KernelA(confA::Int)
    return KernelAstr(confA)
end


function Lux.initialparameters(rng::AbstractRNG, l::KernelAstr)
    return (paramsA=CuArray(rand(rng,Float32, l.confA, l.confA, l.confA)), paramsB = CuArray(rand(rng,Float32, l.confA, l.confA, l.confA)))
end

Lux.initialstates(::AbstractRNG, ::KernelAstr) = NamedTuple()



# # But still recommened to define these
# Lux.parameterlength(l::KernelAstr) = l.out_dims * l.in_dims + l.out_dims

# Lux.statelength(::KernelAstr) = 0

function (l::KernelAstr)(x, ps, st::NamedTuple)
    return call_example_kernel1(x, ps.paramsA),st
end

rng = Random.default_rng()
Nx,Ny,Nz= 64+2,64+2,64+2        

l = KernelA(Nx)

rand(rng, l.confA, l.confA)

ps, st = Lux.setup(rng, l)

println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
        Lux.statelength(l))


x = randn(rng, Float32, Nx, Ny,Nz)
x= CuArray(x)

Lux.apply(l, x, ps, st) # or `l(x, ps, st)`

#x=CuArray(x)


model = Lux.Chain(KernelA(nSize),KernelA(nSize) )
opt = Optimisers.Adam(0.03)

"""
extremely simple loss function - that just wan to decrese sumof all inputs
"""
function loss_function(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    return sum(y_pred), st, ()
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

tstate = main(tstate, vjp_rule, x,250)
y_pred = Lux.cpu(Lux.apply(tstate.model, Lux.gpu(x), tstate.parameters, tstate.states)[1])










# function generate_data(rng::Random.AbstractRNG)
#     x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
#     y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, (1, 128)) .* 0.1f0
#     return (x, y)
# end

# rng = Random.MersenneTwister()
# Random.seed!(rng, 12345)

# (x, y) = generate_data(rng)