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
using Revise


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
end#testKern