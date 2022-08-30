# based on https://github.com/JuliaDiff/ChainRules.jl/issues/665
# abstract diff https://frankschae.github.io/post/abstract_differentiation/
#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
# backpropagation checkpointing https://fluxml.ai/Zygote.jl/dev/adjoints/#Checkpointing-1


using ChainRulesCore,Zygote,CUDA,Enzyme
using CUDAKernels
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays


# Nx, Ny, Nz = 8, 8, 8
# oneSidePad = 1
# totalPad=oneSidePad*2
# A = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
# dA= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

# Aoutout = CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
# dAoutout= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

# p = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
# dp= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

# threads = (4, 4, 4)
# blocks = (2, 2, 2)

function testKern(A, p, Aout)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    Aout[x, y, z] = A[x, y, z] *p[x, y, z] *p[x, y, z] *p[x, y, z] 
    
    return nothing
end

function testKernDeff( A, dA, p
    , dp, Aout
    , dAout)
    Enzyme.autodiff_deferred(testKern, Const, Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout)
    )
    return nothing
end

function calltestKern(A, p)
    Aout = CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
    @cuda threads = threads blocks = blocks testKern( A, p,  Aout)
    return Aout
end

aa=calltestKern(A, p)
maximum(aa)

# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(calltestKern), A, p)
    Aout = calltestKern(A, p)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
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
ress=Zygote.jacobian(calltestKern,A, p )
typeof(ress)
maximum(ress[1])
maximum(ress[2])


#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct KernelAstr<: Lux.AbstractExplicitLayer
    confA::Int
end

function KernelA(confA::Int)
    return KernelAstr(confA)
end


function Lux.initialparameters(rng::AbstractRNG, l::KernelAstr)
    return (paramsA=CuArray(rand(rng,Float32, l.confA, l.confA, l.confA))
    ,paramsB = CuArray(rand(rng,Float32, l.confA, l.confA, l.confA)))
end

Lux.initialstates(::AbstractRNG, ::KernelAstr) = NamedTuple()


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
