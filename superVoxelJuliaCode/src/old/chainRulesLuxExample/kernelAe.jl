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

#`/usr/local/share/julia/environments/v1.10/Manifest.toml`
#add ChainRulesCore,Zygote,Enzyme,KernelAbstractions,Lux,NNlib,Optimisers,Plots,Random,Statistics,FillArrays,LuxCUDA,CUDA,MedEye3d,Images,ImageFiltering
#### test data
Nx, Ny, Nz = 8, 8, 8
oneSidePad = 1
totalPad=oneSidePad*2
A = CUDA.ones(1,3,Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dA= CUDA.ones(1,3,Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

# Aoutout = CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
# dAoutout= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

p = CUDA.ones(3) 
# p = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dp= CUDA.ones(3 ) 
# dp= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

threads = (2,2,2)
blocks = (1, 1, 1)
rng = Random.default_rng()

function sigmoid(x::Float32)::Float32
    return 1 / (1 + exp(-x))
end

#### main kernel
function testKern(prim_A,A, p, Aout,Nx)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) 
    # Aout[x, y, z] = A[x, y, z] *p[x, y, z] *p[x, y, z] *p[x, y, z] 
    # Int(round(sigmoid(p[1]))*Nx),sigmoid(p[2])*Nx,sigmoid(p[3])*Nx

    # Aout[x, y, z] = A[sigmoid(p[1])*Nx,sigmoid(p[2])*Nx,sigmoid(p[3])*Nx]#A[x, y, z] *p[x, y, z] *p[x, y, z] *p[x, y, z] 
    # Aout[x, y, z] = A[Int(round(sigmoid(p[1]))*(Nx-1))+1,Int(round(sigmoid(p[2]))*(Nx-1))+1,Int(round(sigmoid(p[3]))*(Nx-1))+1]#A[x, y, z] *p[x, y, z] *p[x, y, z] *p[x, y, z] 
    # Aout[x*4+y*2+z,1] =Int(round(sigmoid(p[1,x,y,z]))*(Nx-1))+1 
    # Aout[x*4+y*2+z,2] =Int(round(sigmoid(p[2,x,y,z]))*(Nx-1))+1 
    # Aout[x*4+y*2+z,3] =Int(round(sigmoid(p[3,x,y,z]))*(Nx-1))+1
    x_p=Int(round(sigmoid(A[x,y,z,1,1]))*(Nx-3))+2 
    y_p =Int(round(sigmoid(A[x,y,z,2,1]))*(Nx-3))+2 
    z_p =Int(round(sigmoid(A[x,y,z,3,1]))*(Nx-3))+2

    


    # Aout[x*4+y*2+z] = A[x_p-1,y_p,z_p] - A[x_p+1,y_p,z_p] + A[x_p,y_p-1,z_p] - A[x_p,y_p+1,z_p] + A[x_p,y_p,z_p-1] - A[x_p,y_p,z_p+1]
    res=prim_A[x_p-1,y_p,z_p] - prim_A[x_p+1,y_p,z_p] + prim_A[x_p,y_p-1,z_p] - prim_A[x_p,y_p+1,z_p] + prim_A[x_p,y_p,z_p-1] - prim_A[x_p,y_p,z_p+1]
    # Aout[x*4+y*2+z]=res
    Aout[(x-1)*4+(y-1)*2+z]=res
    
    #A[x_p-1,y_p,z_p] - A[x_p+1,y_p,z_p] + A[x_p,y_p-1,z_p] - A[x_p,y_p+1,z_p] + A[x_p,y_p,z_p-1] - A[x_p,y_p,z_p+1]

    # A[Int(round(sigmoid(p[1]))*(Nx-1))+1,Int(round(sigmoid(p[2]))*(Nx-1))+1,Int(round(sigmoid(p[3]))*(Nx-1))+1]#A[x, y, z] *p[x, y, z] *p[x, y, z] *p[x, y, z] 
    
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
    # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
    return Aout
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(calltestKern),prim_A, A, p,Nx)
    

    Aout = calltestKern(prim_A,A, p,Nx)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)
        threads = (2, 2,2)
        blocks = (1, 1, 1)

        dp = CUDA.ones(size(p))
        dprim_A = CUDA.ones(size(prim_A))
        dA = CUDA.ones(size(A))
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)
        @cuda threads = threads blocks = blocks testKernDeff(prim_A,dprim_A, A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)

        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp
        
        return dprim_A,f̄, x̄, ȳ,NoTangent()
    end   
    return Aout, call_test_kernel1_pullback

end

# using Pkg
# Pkg.test("CUDA")
# paramsA = tstate.parameters.paramsA
# println(fieldnames(typeof(tstate)))
#first testing does custom backpropagation compiles
# ress=Zygote.jacobian(calltestKern,A,p,Nx )


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
"""
https://stackoverflow.com/questions/52035775/in-julia-1-0-how-to-set-a-named-tuple-with-only-one-key-value-pair
in order to get named tuple with single element put comma after
"""
function Lux.initialstates(::AbstractRNG, l::KernelAstr)::NamedTuple
    return (NxSt=l.confA , )
end

function (l::KernelAstr)(x, ps, st::NamedTuple)
    x,prim_a= x
    return calltestKern(prim_a,x, ps.paramsA,ps.Nx),st
end



l = KernelA(Nx)
ps, st = Lux.setup(rng, l)
# println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
#         Lux.statelength(l))

arr = collect(range(1, stop = Nx*Ny*Nz))
arr = reshape(arr, (Nx, Ny, Nz))
arr = Float32.(arr)
arr.=1.0
arr[1:4,1:4,1:4].=0.0

dev = gpu_device()

conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=2, pad=Lux.SamePad())

# model = Lux.Chain(KernelA(Nx),KernelA(Nx)) 
function connection_before_kernelA(x,y)
    return (x,y)
end

model = Lux.Chain(SkipConnection(Lux.Chain(conv1(1,3),conv2(3,3),conv2(3,3)) , connection_before_kernelA; name="prim_convs"),KernelA(Nx)) 

# ,KernelA(Nx)
# SkipConnection(Lux.Chain(conv1(1,3),conv2(3,3),conv2(3,3)) , connection; name=nothing)


# model = Lux.Chain(conv1(1,3)) 
arr_new=arr
for i in range(1,3)
    arr_new=map(el-> imfilter(arr_new[el,:,:], Kernel.gaussian(3)) ,range(1, stop = Nx))
    arr_new=stack(arr_new)

    # arr_new=map(el-> imfilter(arr_new[:,el,:], Kernel.gaussian(3)) ,range(1, stop = Nx))
    # arr_new=stack(arr_new;dims=2)    
    
    # arr_new=map(el-> imfilter(arr_new[:,:,el], Kernel.gaussian(3)) ,range(1, stop = Nx))
    # arr_new=stack(arr_new;dims=3)
end    



arr=reshape(arr_new,(Nx,Ny,Nz,1,1))
arr=Float32.(arr)

# x = randn(rng, Float32, Nx, Ny,Nz)
x = arr
x= CuArray(x)
A=x


# ...

# Apply Gaussian smoothing to A



ps, st = Lux.setup(rng, model) .|> dev
opt = Optimisers.Adam(0.03)
opt_st = Optimisers.setup(opt, ps) |> dev
# vjp_rule = Lux.Training.ZygoteVJP()
vjp_rule = Lux.Training.AutoZygote()

# testing weather forward pass runs
# y_pred, st =Lux.apply(l, x, ps, st)


# opt = Optimisers.Adam(0.0003)

y_pred, st = Lux.apply(model, x, ps, st)

"""
extremely simple loss function we just want to get the result to be as close to 100 as possible
"""
function loss_function(model, ps, st, x)
    # y_pred, st = Lux.apply(model, x, ps, st)
    y_pred, st = Lux.apply(model, x, ps, st)
    
    #y_pred in debug now is 8 cartesian points

    return (sum(y_pred)), st, ()
    # return (loss), st, ()
    # return ([loss]), st,y_pred,()
end

# tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=dev)



function main(ps, st,opt,opt_st , vjp, data,model,
    epochs::Int)
    x = CuArray(data) #.|> Lux.gpu
    for epoch in 1:epochs

        (loss, st), back = Zygote.pullback(p -> loss_function(model, p, st, x), ps)
        gs = back((one(loss), nothing))[1]
        opt_st, ps = Optimisers.update(opt_st, ps, gs)

        # grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
        #                                                         data, tstate)
        @info epoch=epoch loss=loss #tstate=tstate.parameters.paramsA
        # tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return ps, st,opt,opt_st 
end
# one epoch just to check if it runs
ps, st,opt,opt_st  = main(ps, st,opt,opt_st , vjp_rule, x,model,1)
#training 
ps, st,opt,opt_st  = main(ps, st,opt,opt_st , vjp_rule, x,model,30)


