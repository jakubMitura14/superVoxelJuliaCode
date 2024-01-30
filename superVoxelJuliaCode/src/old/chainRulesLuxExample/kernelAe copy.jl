# based on https://github.com/JuliaDiff/ChainRules.jl/issues/665
# abstract diff https://frankschae.github.io/post/abstract_differentiation/
#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
# backpropagation checkpointing https://fluxml.ai/Zygote.jl/dev/adjoints/#Checkpointing-1

using Pkg
using ChainRulesCore,Zygote,CUDA,Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux,LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays
using LinearAlgebra
using Images,ImageFiltering
Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")

#### test data
Nx, Ny, Nz = 8, 8, 8
threads = (2,2,2)
blocks = (1, 1, 1)
rng = Random.default_rng()

function sigmoid(x::Float32)::Float32
    return 1 / (1 + exp(-x))
end

#### main dummy kernel
function testKern(prim_A,A, p, Aout,Nx)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) 

    Aout[(x-1)*4+(y-1)*2+z]=(ceil(A[x,y,z]))

    return nothing
end

function testKernDeff( prim_A,dprim_A,A, dA, p
    , dp, Aout
    , dAout,Nx)
    Enzyme.autodiff_deferred(Reverse,testKern, Const, Duplicated(prim_A, dprim_A),Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout),Const(Nx) )
    return nothing
end


function calltestKern(prim_A,A, p,Nx)
    Aout = CUDA.zeros(Float32,8) 
    @cuda threads = threads blocks = blocks testKern(prim_A, A, p,  Aout,Nx)
    return Aout
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(calltestKern),prim_A, A, p,Nx)
    

    Aout = calltestKern(prim_A,A, p,Nx)
    function call_test_kernel1_pullback(dAout)
        threads = (2, 2,2)
        blocks = (1, 1, 1)

        dp = CUDA.ones(size(p))
        dprim_A = CUDA.ones(size(prim_A))
        dA = CUDA.ones(size(A))
        @cuda threads = threads blocks = blocks testKernDeff(prim_A,dprim_A, A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)

        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp
        
        return dprim_A,f̄, x̄, ȳ,NoTangent()
    end   
    return Aout, call_test_kernel1_pullback

end


#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct KernelAstr<: Lux.AbstractExplicitLayer
    confA::Int
end

function KernelA(confA)
    return KernelAstr(confA)
end

function Lux.initialparameters(rng::AbstractRNG, l::KernelAstr)
    return (paramsA=CuArray(rand(rng,Float32, 3,8))
    ,Nx =l.confA )
end

function Lux.initialstates(::AbstractRNG, l::KernelAstr)::NamedTuple
    return (NxSt=l.confA , )
end

function (l::KernelAstr)(x, ps, st::NamedTuple)
    x,prim_a= x
    return calltestKern(prim_a,x, ps.paramsA,ps.Nx),st
end




conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=2, pad=Lux.SamePad())

# model = Lux.Chain(KernelA(Nx),KernelA(Nx)) 
function connection_before_kernelA(x,y)
    return (x,y)
end


arr = collect(range(1, stop = Nx*Ny*Nz))
arr=reshape(arr,(Nx,Ny,Nz,1,1))
arr=Float32.(arr)
x = arr
x= CuArray(x)

dev = gpu_device()
model = Lux.Chain(SkipConnection(Lux.Chain(conv1(1,3),conv2(3,3),conv2(3,3)) , connection_before_kernelA; name="prim_convs"),KernelA(Nx)) 

ps, st = Lux.setup(rng, model) .|> dev
opt = Optimisers.Adam(0.03)
opt_st = Optimisers.setup(opt, ps) |> dev
vjp_rule = Lux.Training.AutoZygote()
y_pred, st = Lux.apply(model, x, ps, st)

"""
extremely simple loss function we just want to get the result to be as close to 100 as possible
"""
function loss_function(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    return (sum(y_pred)), st, ()

end

function main(ps, st,opt,opt_st , vjp, data,model,
    epochs::Int)
    x = CuArray(data) #.|> Lux.gpu
    for epoch in 1:epochs

        (loss, st), back = Zygote.pullback(p -> loss_function(model, p, st, x), ps)
        gs = back((one(loss), nothing))[1]
        opt_st, ps = Optimisers.update(opt_st, ps, gs)

        @info epoch=epoch loss=loss 
    end
    return ps, st,opt,opt_st 
end
# one epoch just to check if it runs
ps, st,opt,opt_st  = main(ps, st,opt,opt_st , vjp_rule, x,model,1)


