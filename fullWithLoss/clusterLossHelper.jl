


"""
main function used in calculating the loss - more in clustering loss .jl
"""
function lossPrepare_execute(image,out)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #CUDA.@cuprint("")
    #this basically threshold for values below and above 0.5
    softThreshold_half(image[x,y,z,1,1])

    ((alaMax(Float32(p[x,y,z]),Float32(0.5))-0.48)/0.52)

    return nothing
end




"""
Enzyme definitions for calculating derivatives of applyGaussKern in back propagation
"""
function applyGaussKern_Deff(means,d_means,stdGaus,d_stdGaus,origArr
    ,d_origArr,out,d_out,meansLength)
    
    Enzyme.autodiff_deferred(applyGaussKern, Const
    ,Duplicated(means, d_means)
    ,Duplicated(stdGaus,d_stdGaus)
    ,Duplicated(origArr, d_origArr)
    ,Duplicated(out, d_out)
    ,Const(meansLength)    )
    return nothing
end


"""
call function with out variable initialization
"""
function callGaussApplyKern(means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss)
    out = CUDA.zeros(size(origArr)) 
    @cuda threads = threads_apply_gauss blocks = blocks_apply_gauss applyGaussKern(means,stdGaus,origArr,out,meansLength)
    return out
end

# aa=calltestKern(A, p)
# maximum(aa)

# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(callGaussApplyKern),means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss)
    out = callGaussApplyKern(means,stdGaus,origArr,meansLength,threads_apply_gauss,blocks_apply_gauss)
    function call_test_kernel1_pullback(d_out_prim)
        # Allocate shadow memory.
        d_means = CUDA.ones(size(means))
        d_stdGaus = CUDA.ones(size(stdGaus))
        d_origArr = CUDA.ones(size(origArr))
        d_out = CuArray(collect(d_out_prim))
        @cuda threads = threads_apply_gauss blocks = blocks_apply_gauss applyGaussKern_Deff(means,d_means,stdGaus,d_stdGaus,origArr,d_origArr,out,d_out,meansLength)
        f̄ = NoTangent()

        return f̄, d_means,d_stdGaus, d_origArr, NoTangent(), NoTangent(), NoTangent()
    end   
    return out, call_test_kernel1_pullback

end
