using ChainRulesCore
using CUDA
using CUDAKernels
using Enzyme
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random




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
    event = kernel(x, out, p, ndrange=size(x))
    wait(event)
    return z
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
        event = gpu_kernel_autodiff(dx, dy, dz, ndrange=4)
        
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

function (l::KernelAstr)(x::AbstractMatrix, ps, st::NamedTuple)
    call_example_kernel1(x, ps.paramsA)
    return out
end


rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, l)