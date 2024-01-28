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
using Revise
using Images,ImageFiltering
includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern.jl")

struct KernelAstr<: Lux.AbstractExplicitLayer
    Nx::Int
    threads::Tuple{Int,Int,Int}
    blocks::Tuple{Int,Int,Int}
end

function KernelA(Nx,threads,blocks)
    return KernelAstr(Nx,threads,blocks)
end

function Lux.initialparameters(rng::AbstractRNG, l::KernelAstr)
    return (paramsA=CuArray(rand(rng,Float32, 3,8))
    ,Nx =l.Nx,threads=l.threads,blocks=l.blocks )
end
"""
https://stackoverflow.com/questions/52035775/in-julia-1-0-how-to-set-a-named-tuple-with-only-one-key-value-pair
in order to get named tuple with single element put comma after
"""
function Lux.initialstates(::AbstractRNG, l::KernelAstr)::NamedTuple
    return (NxSt=l.Nx , )
end

function (l::KernelAstr)(x, ps, st::NamedTuple)
    x,prim_a= x
    return calltestKern(prim_a,x, ps.paramsA,ps.Nx,ps.threads,ps.blocks),st
end



#get convolutions
conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=2, pad=Lux.SamePad())

# model = Lux.Chain(KernelA(Nx),KernelA(Nx)) 
function connection_before_kernelA(x,y)
    return (x,y)
end


function get_model_consts(dev,Nx,threads,blocks)
    model = Lux.Chain(SkipConnection(Lux.Chain(conv1(1,3),conv2(3,3),conv2(3,3)) , connection_before_kernelA; name="prim_convs"),KernelA(Nx,threads,blocks)) 
    ps, st = Lux.setup(rng, model) .|> dev
    opt = Optimisers.Adam(0.003)
    opt_st = Optimisers.setup(opt, ps) |> dev
    # vjp_rule = Lux.Training.ZygoteVJP()
    vjp_rule = Lux.Training.AutoZygote()

return model,ps, st,opt,opt_st,vjp_rule
end#get_model_consts