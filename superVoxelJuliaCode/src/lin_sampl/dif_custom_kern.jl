using Pkg
using ChainRulesCore,Zygote,CUDA,Enzyme
using KernelAbstractions
using Zygote, Lux,LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays
using LinearAlgebra
using Images,ImageFiltering
using Revise
includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")

function testKernDeff( prim_A,dprim_A,A, dA, p
    , dp, Aout
    , dAout,Nx)
    Enzyme.autodiff_deferred(Reverse,testKern, Const, Duplicated(prim_A, dprim_A),Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout),Const(Nx) )
    return nothing
end


function calltestKern(prim_A,A, p,Nx,threads,blocks)
    Aout = CUDA.zeros(Float32,8) 
    @cuda threads = threads blocks = blocks testKern(prim_A, A, p,  Aout,Nx)
    # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
    return Aout
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(calltestKern),prim_A, A, p,Nx,threads,blocks)
    

    Aout = calltestKern(prim_A,A, p,Nx,threads,blocks)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)


        dp = CUDA.ones(size(p))
        dprim_A = CUDA.ones(size(prim_A))
        dA = CUDA.ones(size(A))
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)
        @cuda threads = threads blocks = blocks testKernDeff(prim_A,dprim_A, A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)

        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp
        
        return dprim_A,f̄, x̄, ȳ,NoTangent(),NoTangent(),NoTangent()
    end   
    return Aout, call_test_kernel1_pullback

end