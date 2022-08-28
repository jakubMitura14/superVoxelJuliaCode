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

# Two kernels to be called one after the other.
@kernel function example_kernel(x,out, p)
    i, j = @index(Global, NTuple)
    out[i,j]=x[i,j]*p[i,j]
    nothing
end

# Function calls to allow easier high-level code.
function call_example_kernel1(x, p,out)
    # out=ones(Float32,size(x))
    #kernel = example_kernel(CUDADevice())
    kernel = example_kernel(CPU())
    event = kernel(x, out, p, ndrange=size(x))
    wait(event)
    return nothing
    #return out
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_example_kernel1), x, p)
    z=ones(Float32,size(x))
    call_example_kernel1(x, p,z)

    function call_example_kernel1_pullback(z̄)
        # Allocate shadow memory.
        dz_dx = zeros(Float32,size(x))
        dz_dp = zeros(Float32,size(p))

        # Define differentials.
        dx = Duplicated(x, dz_dx)
        dp = Duplicated(p, dz_dp)
        dz = Duplicated(z, collect(z̄))
    
        # AD call.
        # gpu_kernel_autodiff = autodiff(example_kernel(CUDADevice()))
        # gpu_kernel_autodiff = autodiff(call_example_kernel1(CPU()))
        # event = gpu_kernel_autodiff(dx, dp, dz, ndrange=size(x))
        Enzyme.autodiff_deferred(call_example_kernel1, Const,dx, dp, dz)
        # Return differentials of input.
        f̄ = NoTangent()
        x̄ = dx.dval
        ȳ = dy.dval
        
        return f̄, x̄, ȳ
    end
    
    return z, call_example_kernel1_pullback
end




#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct KernelAstr<: Lux.AbstractExplicitLayer
    confA::Int
end

function KernelA(confA::Int)
    return KernelAstr(confA)
end


function Lux.initialparameters(rng::AbstractRNG, l::KernelAstr)
    return (paramsA=rand(rng,Float32, l.confA, l.confA), paramsB = rand(rng,Float32, l.confA, l.confA))
end

Lux.initialstates(::AbstractRNG, ::KernelAstr) = NamedTuple()



# # But still recommened to define these
# Lux.parameterlength(l::KernelAstr) = l.out_dims * l.in_dims + l.out_dims

# Lux.statelength(::KernelAstr) = 0

function (l::KernelAstr)(x::AbstractMatrix, ps, st::NamedTuple)
    z=ones(Float32,size(x))
    return call_example_kernel1(x, ps.paramsA,z),st
end

rng = Random.default_rng()
l = KernelA(nSize)

rand(rng, l.confA, l.confA)

ps, st = Lux.setup(rng, l)

println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
        Lux.statelength(l))

x = randn(rng, Float32, nSize, nSize)

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

# tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
tstate = Lux.Training.TrainState(rng, model, opt)
vjp_rule = Lux.Training.ZygoteVJP()


function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data::AbstractMatrix,
    epochs::Int)
    #data = data .|> Lux.gpu
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