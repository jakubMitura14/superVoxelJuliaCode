using ChainRulesCore
using CUDA
using CUDAKernels
using Enzyme
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote


nSize = 3

# Two kernels to be called one after the other.
@kernel function example_kernel(x,out, p)
    i, j = @index(Global, NTuple)
    out[i,j]=x[i,j]*p[i,j]
    nothing
end

# Function calls to allow easier high-level code.
function call_example_kernel1(x, p)
    out=similar(x)
    fill!(out, 1)
    kernel = example_kernel(CUDADevice())
    #kernel = example_kernel(CPU())
    event = kernel(x, out, p, ndrange=size(x))
    wait(event)
    return out
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_example_kernel1), x, y)
    z = call_example_kernel1(x, y)

    function call_example_kernel1_pullback(z̄)
        # Allocate shadow memory.
        dz_dx = similar(x)
        fill!(dz_dx, 0)
        dz_dy = similar(y)
        fill!(dz_dy, 0)

        # Define differentials.
        dx = Duplicated(x, dz_dx)
        dy = Duplicated(y, dz_dy)
        dz = Duplicated(z, z̄)
    
        # AD call.
        gpu_kernel_autodiff = autodiff(example_kernel(CUDADevice()))
        event = gpu_kernel_autodiff(dx, dy, dz, ndrange=size(x))
        
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
    return (paramsA=rand(rng, l.confA, l.confA), paramsB = rand(rng, l.confA, l.confA))
end

Lux.initialstates(::AbstractRNG, ::KernelAstr) = NamedTuple()



# # But still recommened to define these
# Lux.parameterlength(l::KernelAstr) = l.out_dims * l.in_dims + l.out_dims

# Lux.statelength(::KernelAstr) = 0

function (l::KernelAstr)(x::AbstractMatrix, ps, st::NamedTuple)
    return call_example_kernel1(x, ps.paramsA)
end

rng = Random.default_rng()
l = KernelA(nSize)

rand(rng, l.confA, l.confA)

ps, st = Lux.setup(rng, l)

println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
        Lux.statelength(l))

x = randn(rng, Float32, nSize, nSize)
x=CuArray(x)

Lux.apply(l, x, ps, st) # or `l(x, ps, st)`



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
vjp_rule = Lux.Training.ZygoteVJP()


function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data::AbstractMatrix,
    epochs::Int)
    data = data .|> Lux.gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
                                                                data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

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