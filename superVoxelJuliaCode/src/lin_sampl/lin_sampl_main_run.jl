using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux, LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra

using Revise
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/Lux_model.jl")

Nx, Ny, Nz = 8, 8, 8
rng = Random.default_rng()
threads = (2, 2, 2)
blocks = (1, 1, 1)
dev = gpu_device()

function get_sample_dat()
    arr = collect(range(1, stop=Nx * Ny * Nz))
    arr = reshape(arr, (Nx, Ny, Nz))
    arr = Float32.(arr)
    arr .= 1.0
    arr[1:4, 1:4, 1:4] .= 0.0

    arr_new = arr
    for i in range(1, 3)
        arr_new = map(el -> imfilter(arr_new[el, :, :], Kernel.gaussian(3)), range(1, stop=Nx))
        arr_new = stack(arr_new)

        # arr_new=map(el-> imfilter(arr_new[:,el,:], Kernel.gaussian(3)) ,range(1, stop = Nx))
        # arr_new=stack(arr_new;dims=2)    

        # arr_new=map(el-> imfilter(arr_new[:,:,el], Kernel.gaussian(3)) ,range(1, stop = Nx))
        # arr_new=stack(arr_new;dims=3)
    end

    arr = reshape(arr_new, (Nx, Ny, Nz, 1, 1))
    arr = Float32.(arr)

    # x = randn(rng, Float32, Nx, Ny,Nz)
    x = arr
    x = CuArray(x)
    return x
end

x = get_sample_dat()


function loss_function(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    return (sum(y_pred)), st, ()
end


function main(ps, st, opt, opt_st, vjp, data, model,
    epochs::Int)
    x = CuArray(data) #.|> Lux.gpu
    for epoch in 1:epochs

        (loss, st), back = Zygote.pullback(p -> loss_function(model, p, st, x), ps)
        gs = back((one(loss), nothing))[1]
        opt_st, ps = Optimisers.update(opt_st, ps, gs)

        # grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
        #                                                         data, tstate)
        @info epoch = epoch loss = loss #tstate=tstate.parameters.paramsA
        # tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return ps, st, opt, opt_st
end

#initialization
model, ps, st, opt, opt_st, vjp_rule = get_model_consts(dev, Nx, threads, blocks)

# y_pred, st = Lux.apply(model, x, ps, st)

# one epoch just to check if it runs
ps, st, opt, opt_st = main(ps, st, opt, opt_st, vjp_rule, x, model, 1)

#training 
ps, st, opt, opt_st = main(ps, st, opt, opt_st, vjp_rule, x, model, 300)

