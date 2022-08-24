using ChainRulesCore
using CUDA
using CUDAKernels
using Enzyme
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random

#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct KernelA{F1, F2} <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
end

function KernelA(in_dims::Int, out_dims::Int; init_weight=Lux.glorot_uniform,
                init_bias=Lux.zeros32)
    return KernelA{typeof(init_weight), typeof(init_bias)}(in_dims, out_dims, init_weight,
                                                          init_bias)
end

l = KernelA(2, 4)
function Lux.initialparameters(rng::AbstractRNG, l::KernelA)
    return (weight=l.init_weight(rng, l.out_dims, l.in_dims),
            bias=l.init_bias(rng, l.out_dims, 1))
end

Lux.initialstates(::AbstractRNG, ::KernelA) = NamedTuple()

# But still recommened to define these
Lux.parameterlength(l::KernelA) = l.out_dims * l.in_dims + l.out_dims

Lux.statelength(::KernelA) = 0

function (l::KernelA)(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * x .+ ps.bias
    return y, st
end


rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, l)