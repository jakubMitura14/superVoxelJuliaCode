# based on https://github.com/JuliaDiff/ChainRules.jl/issues/665
# abstract diff https://frankschae.github.io/post/abstract_differentiation/
#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
# backpropagation checkpointing https://fluxml.ai/Zygote.jl/dev/adjoints/#Checkpointing-1

using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux, LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra

# Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")
using CUDA, Enzyme, Random
Enzyme.API.printall!(true)
Nx, Ny, Nz = 8, 8, 8
threads = (2, 2, 2)
blocks = (1, 1, 1)

#### main dummy kernel
function testKern(prim_A, A, p, Aout, Nx)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y()))
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z()))

    Aout[(x-1)*4+(y-1)*2+z] = (ceil(A[x, y, z]))

    return nothing
end

function testKernDeff(prim_A, dprim_A, A, dA, p, dp, Aout, dAout, Nx)
    Enzyme.autodiff_deferred(Reverse, testKern, Const, Duplicated(prim_A, dprim_A), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout), Const(Nx))
    return nothing
end


function calltestKern(prim_A, A, p, Nx)
    Aout = CUDA.zeros(Float32, 8)
    @cuda threads = threads blocks = blocks testKern(prim_A, A, p, Aout, Nx)
    return Aout
end



prim_A = CUDA.zeros(Float32, Nx, Ny, Nz)
dprim_A = CUDA.zeros(Float32, Nx, Ny, Nz)
A = CUDA.zeros(Float32, Nx, Ny, Nz)
dA = CUDA.zeros(Float32, Nx, Ny, Nz)

p = CUDA.zeros(Float32, Nx, Ny, Nz)
dp = CUDA.zeros(Float32, Nx, Ny, Nz)
Aout = CUDA.zeros(Float32, 8)
dAout = CUDA.zeros(Float32, 8)

@cuda threads = threads blocks = blocks testKernDeff(prim_A, dprim_A, A, dA, p, dp, Aout, dAout, Nx)