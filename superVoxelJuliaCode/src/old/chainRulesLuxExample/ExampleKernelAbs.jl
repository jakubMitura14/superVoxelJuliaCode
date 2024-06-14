# based on https://github.com/JuliaDiff/ChainRules.jl/issues/665
# abstract diff https://frankschae.github.io/post/abstract_differentiation/
#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
# backpropagation checkpointing https://fluxml.ai/Zygote.jl/dev/adjoints/#Checkpointing-1


using ChainRulesCore, Zygote, CUDA, Enzyme
using CUDAKernels
using KernelAbstractions
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote, Test


@kernel function matmul_kernel!(a, b, c)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for k = 1:size(a)[2]
        @inbounds tmp_sum += a[i, k] * b[k, j]
    end

    c[i, j] = tmp_sum
end


function matmul_testsuite(backend, ArrayT)

    matmul = matmul_kernel!(backend(), (32, 32))
    a = ArrayT(rand(32, 64))
    b = ArrayT(rand(64, 32))
    c = ArrayT(zeros(32, 32))
    # a = ArrayT(rand(128, 256))
    # b = ArrayT(rand(256, 128))
    # c = ArrayT(zeros(128, 128))
    wait(matmul(a, b, c, ndrange=size(c)))

    @test c ≈ a * b

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

    @test da ≈ compare_dc * b'
    @test db ≈ a' * compare_dc
end


function testsuite(backend, backend_str, backend_mod, AT, DAT)
    @testset "Matmul" begin
        matmul_testsuite(backend, AT)
    end
end


struct CPUDeviceArray{T,N,A} end # Fake and unused
testsuite(CPU, "CPU", Base, Array, CPUDeviceArray)

testsuite(CUDADevice, "CUDA", CUDA, CuArray, CUDA.CuDeviceArray)

















#### test data
Nx, Ny, Nz = 8, 8, 8
oneSidePad = 1
totalPad = oneSidePad * 2
A = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)
dA = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)

Aoutout = CUDA.zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad)
dAoutout = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)

p = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)
dp = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)

threads = (4, 4, 4)
blocks = (2, 2, 2)
rng = Random.default_rng()

#### main kernel
function testKern(A, p, Aout, Nx)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    Aout[x, y, z] = A[x, y, z] * p[x, y, z] * p[x, y, z] * p[x, y, z]

    return nothing
end

function testKernDeff(A, dA, p, dp, Aout, dAout, Nx)
    Enzyme.autodiff_deferred(testKern, Const, Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout), Const(Nx)
    )
    return nothing
end

function calltestKern(A, p, Nx)
    Aout = CUDA.zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad)
    @cuda threads = threads blocks = blocks testKern(A, p, Aout, Nx)
    return Aout
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(calltestKern), A, p, Nx)

    Aout = calltestKern(A, p, Nx)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)
        threads = (4, 4, 4)
        blocks = (2, 2, 2)
        dp = CUDA.ones(size(p))
        dA = CUDA.ones(size(A))
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)
        @cuda threads = threads blocks = blocks testKernDeff(A, dA, p, dp, Aout, CuArray(collect(dAout)), Nx)

        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp

        return f̄, x̄, ȳ, NoTangent()
    end
    return Aout, call_test_kernel1_pullback

end


#first testing does custom backpropagation compiles
ress = Zygote.jacobian(calltestKern, A, p, Nx)


#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct KernelAstr <: Lux.AbstractExplicitLayer
    confA::Int
end

function KernelA(confA)
    return KernelAstr(confA)
end

function Lux.initialparameters(rng::AbstractRNG, l::KernelAstr)
    return (paramsA=CuArray(rand(rng, Float32, l.confA, l.confA, l.confA)), Nx=l.confA)
end
"""
https://stackoverflow.com/questions/52035775/in-julia-1-0-how-to-set-a-named-tuple-with-only-one-key-value-pair
in order to get named tuple with single element put comma after
"""
function Lux.initialstates(::AbstractRNG, l::KernelAstr)::NamedTuple
    return (NxSt=l.confA,)
end

function (l::KernelAstr)(x, ps, st::NamedTuple)
    return calltestKern(x, ps.paramsA, ps.Nx), st
end



l = KernelA(Nx)
ps, st = Lux.setup(rng, l)
println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
    Lux.statelength(l))

x = randn(rng, Float32, Nx, Ny, Nz)
x = CuArray(x)
# testing weather forward pass runs
y_pred, st = Lux.apply(l, x, ps, st)



model = Lux.Chain(KernelA(Nx), KernelA(Nx))
opt = Optimisers.Adam(0.0003)

"""
extremely simple loss function we just want to get the result to be as close to 100 as possible
"""
function loss_function(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    return (100 - sum(y_pred))^2, st, ()
end

tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
vjp_rule = Lux.Training.ZygoteVJP()


function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data,
    epochs::Int)
    # data = data .|> Lux.gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
            data, tstate)
        @info epoch = epoch loss = loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end
# one epoch just to check if it runs
tstate = main(tstate, vjp_rule, x, 1)
#training 
tstate = main(tstate, vjp_rule, x, 1000)

# using ChainRulesTestUtils
# test_rrule(testKern,A, p, Aout)

# a=Fill(7.0f0, 3, 2)
# collect(a)


# using ChainRulesCore,Zygote,CUDA,Enzyme
# using CUDAKernels
# using KernelAbstractions
# using Zygote, Lux
# using Lux, Random
# import NNlib, Optimisers, Plots, Random, Statistics, Zygote
# 

# mkpath("superVoxel_Julia")
# cd("my_image_name")

# pkgs = [
#     "Foo", # Replace Foo, Bar, Baz, etc. with the names of actual packages that you want to use
#     "Bar",
#     "Baz",
# ]
# julia_version = v"1.4.0"

# SimpleContainerGenerator.create_dockerfile(pkgs;
#                                            julia_version = julia_version,
#                                            output_directory = pwd())
