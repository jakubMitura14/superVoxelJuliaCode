using ChainRulesCore
using CUDA
using CUDAKernels
using Enzyme
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random

#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct KernelAstr<: Lux.AbstractExplicitLayer
    confA::Int
end

function KernelA(confA::Int)
    return KernelAstr(confA)
end

l = KernelA(2)

function Lux.initialparameters(rng::AbstractRNG, l::KernelAstr)
    return (paramsA=rand(rng, l.confA, l.confA))
end

Lux.initialstates(::AbstractRNG, ::KernelAstr) = NamedTuple()


println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
        Lux.statelength(l))


# # But still recommened to define these
# Lux.parameterlength(l::KernelAstr) = l.out_dims * l.in_dims + l.out_dims

# Lux.statelength(::KernelAstr) = 0

function (l::KernelA)(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * x .+ ps.bias
    return y, st
end


rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, l)