using ChainRulesCore,Zygote,CUDA,Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays

function mul_kernel(A)
    i = threadIdx().x
    if i <= length(A)
        A[i] *= A[i]
    end
    return nothing
end

function grad_mul_kernel(A, dA)
    Enzyme.autodiff_deferred(Reverse, mul_kernel, Const, Duplicated(A, dA))
    return nothing
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(calltestKern), A)
    Aout = mul_kernel(A) #CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
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

ress=Zygote.jacobian(calltestKern,A,p,Nx )


function KernelA(confA::Int,Nxa,Nya,Nza)
    return KernelAstr(confA,Nxa,Nya,Nza)
end


# @testset "mul_kernel" begin
#     A = CUDA.ones(64,)
#     @cuda threads=length(A) mul_kernel(A)
#     A = CUDA.ones(64,)
#     dA = similar(A)
#     dA .= 1
#     @cuda threads=length(A) grad_mul_kernel(A, dA)
#     @test all(dA .== 2)
# end
