using ChainRulesCore
using CUDA
using CUDAKernels
using Enzyme
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays,Test

CUDA.zeros(5,5,5)
# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(a, b, c)
    i, j = @index(Global, NTuple)

    # # creating a temporary sum variable for matrix multiplication
    # tmp_sum = zero(eltype(c))
    # for k = 1:size(a)[2]
    #     @inbounds tmp_sum += a[i,k] * b[k, j]
    # end

    c[i,j] = a[i,j]*b[i,j]
end



# matmul = matmul_kernel!(CUDADevice(), (32, 32))
matmul = matmul_kernel!(CPU(), (32, 32))
a = rand(128, 128)
b = rand(128, 128)
c = zeros(128, 128)
wait(matmul(a, b, c, ndrange=size(c)))


dmatmul = Enzyme.autodiff(matmul)
da = similar(a)
da .= 0
db = similar(b)
db .= 0
dc = similar(c)
dc .= 1
c .= 0

compare_dc = copy(dc)
wait(dmatmul(
    Duplicated(a, da),
    Duplicated(b, db),
    Duplicated(c, dc), ndrange=size(c)))


da
@test da ≈ compare_dc * b'
@test db ≈ a' * compare_dc

matmul_testsuite(CPU)
