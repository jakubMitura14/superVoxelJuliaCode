# based on https://github.com/JuliaDiff/ChainRules.jl/issues/665
# abstract diff https://frankschae.github.io/post/abstract_differentiation/
#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
using ChainRulesCore
using CUDA
using CUDAKernels
using Enzyme
using KernelAbstractions
using KernelGradients
using Zygote, Lux
using Lux, Random



# Two kernels to be called one after the other.
@kernel function example_kernel(x, y, z)
    i = @index(Global)
    if(i == 1)
        z[i] = 2 * x[i] + y[i]
    elseif (i == 2)
        z[i] = 3 * x[i] + y[i]
    elseif (i == 3)
        z[i] = 4 * x[i] + y[i]
    elseif (i == 4)
        z[i] = 5 * x[i] + y[i]
    end
    nothing
end


@kernel function example_kernel2(z, a, result)
    i = @index(Global)
    result[i] = 3 * z[i] + a[i]
    nothing
end


# Function calls to allow easier high-level code.
function call_example_kernel1(x, y)
    z = similar(x)
    fill!(z, 1)

    kernel = example_kernel(CUDADevice())
    event = kernel(x, y, z, ndrange=4)
    wait(event)
    return z
end


function call_example_kernel2(z, a)
    result = similar(x)
    fill!(result, 1)

    kernel = example_kernel2(CUDADevice())
    event = kernel(z, a, result, ndrange=4)
    wait(event)
    return result
end


function call_all(x, y, a)
    z = call_example_kernel1(x, y)
    result = call_example_kernel2(z, a)
    
    return result
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_example_kernel1), x, y)
    z = call_example_kernel1(x, y)

    function call_example_kernel1_pullback(z̄)
        # Allocate shadow memory.
        dz_dx = similar(x)
        fill!(dz_dx, 0)
        dz_dy = similar(y)
        fill!(dz_dy, 0)

        # Define differentials.
        dx = Duplicated(x, dz_dx)
        dy = Duplicated(y, dz_dy)
        dz = Duplicated(z, z̄)
    
        # AD call.
        gpu_kernel_autodiff = autodiff(example_kernel(CUDADevice()))
        event = gpu_kernel_autodiff(dx, dy, dz, ndrange=4)
        
        # Return differentials of input.
        f̄ = NoTangent()
        x̄ = dx.dval
        ȳ = dy.dval
        
        return f̄, x̄, ȳ
    end
    
    return z, call_example_kernel1_pullback
end


function ChainRulesCore.rrule(::typeof(call_example_kernel2), z, a)
    z = call_example_kernel2(z, a)

    function call_example_kernel2_pullback(result_bar)
        # Allocate shadow memory.
        dresult_dz = similar(z)
        fill!(dresult_dz, 0)
        dresult_da = similar(a)
        fill!(dresult_da, 0)

        # Define differentials.
        dz = Duplicated(x, dresult_dz)
        da = Duplicated(y, dresult_da)
        dresult = Duplicated(z, result_bar)
    
        # AD call.
        gpu_kernel_autodiff = autodiff(example_kernel2(CUDADevice()))
        event = gpu_kernel_autodiff(dz, da, dresult, ndrange=4)
        
        # Return differentials of input.
        f̄ = NoTangent()
        z̄ = dz.dval
        ā = da.dval
        
        return f̄, z̄, ā
    end
    
    return z, call_example_kernel2_pullback
end


# Example input.
x = cu([1., 2, 3, 4])
y = cu([5., 6, 7, 8])
a = cu([9., 10, 11, 12])

z = call_example_kernel1(x, y)

# Calculation without gradients:
# call_all(x, y, a)

Jx, Jy = Zygote.jacobian(call_example_kernel1, x, y)

@show Jx;
@show Jy;

Jz, Ja = Zygote.jacobian(call_example_kernel2, z, a)

@show Jz;
@show Ja;

Jx, Jy, Ja = Zygote.jacobian(call_all, x, y, a)

@show Jx;
@show Jy;
@show Jz;