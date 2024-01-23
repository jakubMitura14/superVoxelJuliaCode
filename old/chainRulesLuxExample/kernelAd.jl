# based on https://github.com/JuliaDiff/ChainRules.jl/issues/665
# abstract diff https://frankschae.github.io/post/abstract_differentiation/
#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
# backpropagation checkpointing https://fluxml.ai/Zygote.jl/dev/adjoints/#Checkpointing-1


using ChainRulesCore,Zygote,CUDA,Enzyme
using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays


Nx, Ny, Nz = 8, 8, 8
oneSidePad = 1
totalPad=oneSidePad*2
A = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dA= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

Aoutout = CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dAoutout= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

p = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dp= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

threads = (4, 4, 4)
blocks = (2, 2, 2)



#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct KernelAstr<: Lux.AbstractExplicitLayer
    confA::Int
    Nxa::Int
    Nya::Int
    Nza::Int
end


function testKern(A, p, Aout,Nxa,Nya,Nza)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    Aout[x, y, z] = A[x, y, z] *p[x, y, z] *p[x, y, z] *p[x, y, z] 
    
    return nothing
end

function testKernDeff( A, dA, p
    , dp, Aout
    , dAout,Nxa,Nya,Nza)
    Enzyme.autodiff_deferred(testKern, Const, Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout),Const(Nxa),Const(Nya),Const(Nza))
    return nothing
end

function calltestKern(A, p,Nxa,Nya,Nza)
    Aout = CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
    @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nxa,Nya,Nza)
    return Aout
end

aa=calltestKern(A, p,Nx,Ny,Nz)
maximum(aa)

# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(calltestKern), A, p,Nxa,Nya,Nza)
    Aout = calltestKern(A, p,Nxa,Nya,Nza)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)
        # Allocate shadow memory.
        threads = (4, 4, 4)
        blocks = (2, 2, 2)
        dp = CUDA.ones(size(p))
        dA = CUDA.ones(size(A))
        @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nxa,Nya,Nza)

        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp
        
        return f̄, x̄, ȳ,NoTangent(),NoTangent(),NoTangent()
    end   
    return Aout, call_test_kernel1_pullback

end
#first testing
# ress=Zygote.jacobian(calltestKern,A, p ,Nx,Ny,Nz)
# typeof(ress)
# maximum(ress[1])
# maximum(ress[2])




function KernelA(confA::Int,Nxa,Nya,Nza)
    return KernelAstr(confA,Nxa,Nya,Nza)
end


function Lux.initialparameters(rng::AbstractRNG, l::KernelAstr)
    return (paramsA=CuArray(rand(rng,Float32, l.confA, l.confA, l.confA))
    ,paramsB = CuArray(rand(rng,Float32, l.Nxa, l.Nya, l.Nza)))
end

Lux.initialstates(::AbstractRNG, ::KernelAstr) = (Nxa=Nx,Nya=Ny,Nza=Nz)


# # But still recommened to define these
# Lux.parameterlength(l::KernelAstr) = l.out_dims * l.in_dims + l.out_dims

# Lux.statelength(::KernelAstr) = 0

function (l::KernelAstr)(x, ps, st::NamedTuple)
    return calltestKern(x, ps.paramsA,st.Nxa,st.Nya,st.Nza),st
end

rng = Random.default_rng()
oneSidePad = 1
totalPad=oneSidePad*2  
Nx, Ny, Nz = 8+totalPad, 8+totalPad, 8+totalPad

l = KernelA(Nx,Nx, Ny,Nz)

ps, st = Lux.setup(rng, l)

println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
        Lux.statelength(l))


x = randn(rng, Float32, Nx, Ny,Nz)
x= CuArray(x)

y_pred, st =Lux.apply(l, x, ps, st) # or `l(x, ps, st)`

model = Lux.Chain(KernelA(Nx,Nx, Ny,Nz),KernelA(Nx,Nx, Ny,Nz) )
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
