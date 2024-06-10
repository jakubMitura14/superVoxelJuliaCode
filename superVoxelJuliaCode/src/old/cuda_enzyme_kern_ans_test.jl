using KernelAbstractions
using ChainRulesCore, Zygote, CUDA, Enzyme, Test







@kernel function example_kenr(@Const(A),A_out)

    index = @index(Global)
    shared_arr = @localmem Float32 (@groupsize()[1], 1)
    shared_arr[@index(Local, Linear)] = A[index]
    A_out[index] = shared_arr[@index(Local, Linear), 1]
    index = @index(Global)
end

function call_example(A,A_out)
    dev = get_backend(A)
    example_kenr(dev, 256)(A,A_out, ndrange=(size(A)[1]))
    KernelAbstractions.synchronize(dev)
    return nothing
end


A=CUDA.ones(10).*2
A_out=CUDA.ones(10)
call_example(A,A_out)
@test A_out == CUDA.ones(10).*2


function ChainRulesCore.rrule(::typeof(call_example), A,A_out)

    #modify A_out by mutation
    call_example(A,A_out)

    function call_test_kernel1_pullback(d_A_out)
        d_A_out = CuArray(collect(d_A_out))
        d_A = CUDA.zeros(size(A)...)

        Enzyme.autodiff_deferred(Enzyme.Reverse, call_example, Const, Duplicated(A,d_A), Duplicated(A_out, d_A_out))

        #NoTangent for the function itself
        return NoTangent(), d_A,d_A_out
    end


    return A_out, call_test_kernel1_pullback

end

out,pull_back=rrule(call_example,A,A_out)
pull_back(CUDA.ones(10))

