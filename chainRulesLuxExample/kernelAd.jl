using ChainRulesCore,Zygote,CUDA,Enzyme


# using CUDAKernels
# using KernelAbstractions
# using KernelGradients
# using Zygote, Lux
# using Lux, Random
# import NNlib, Optimisers, Plots, Random, Statistics, Zygote
# using FillArrays


Nx, Ny, Nz = 8, 8, 8
oneSidePad = 1
totalPad=oneSidePad*2
A = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dA= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

Aout = CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dAout= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

p = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
dp= CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 



function testKern(A, p, Aout)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    Aout[x, y, z] = A[x, y, z] *p[x, y, z] *p[x, y, z] *p[x, y, z] 
    
    return nothing
end

function testKernDeff( A, dA, p
    , dp, Aout
    , dAout)
    Enzyme.autodiff_deferred(testKern, Const, Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout)
    )
    return nothing
end


### run
threads = (4, 4, 4)
blocks = (2, 2, 2)
@cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, dAout)
@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, dAout)
@cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, dAout)
maximum(dp)
maximum(dA)




# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(testKern), A, p,Aout)

    function call_test_kernel1_pullback(dAout)
        # Allocate shadow memory.
        threads = (4, 4, 4)
        blocks = (2, 2, 2)
        dp = CUDA.ones(size(p))
        dA = CUDA.ones(size(A))
        @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, collect(dAout))

        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp
        
        return f̄, x̄, ȳ
    end
    
    return Aout, call_test_kernel1_pullback
end

Zygote.jacobian(testKern,A, p,Aout )


# using ChainRulesTestUtils
# test_rrule(testKern,A, p, Aout)

# a=Fill(7.0f0, 3, 2)
# collect(a)
