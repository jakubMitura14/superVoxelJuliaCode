using CUDA, Enzyme, Test
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl



using CUDA
using Enzyme
using Test

function mul_kernel(Nx,Ny,Nz,A,p)
    x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))
    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))
    z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))
   
    A[x,y,z] *= A[x,y,z]
    return nothing
end

function grad_mul_kernel(Nx,Ny,Nz,A, dA,p,dp)
    Enzyme.autodiff_deferred(mul_kernel, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp))
    return nothing
end
Nx,Ny,Nz= 64,64,64
blocks= Int(Nx*Ny*Nz/512)
A = CUDA.ones(Nx,Ny,Nz)
dA = similar(A)
p = CUDA.ones(Nx,Ny,Nz)
dp = similar(A)
dA .= 1
@cuda threads= (8*8*8) blocks=blocks grad_mul_kernel(Nx,Ny,Nz,A, dA,p,dp)
dA