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




@kernel function example_kernel2(z, a, result)
    i = @index(Global)
    result[i] = 3 * z[i] + a[i]
    nothing
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