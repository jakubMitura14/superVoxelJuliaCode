using Revise
# includet("/workspaces/superVoxelJuliaCode/utils/includeAll.jl")
using ChainRulesCore, Zygote, CUDA, Enzyme
using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

"""
utility macro to iterate in given range around given voxel
"""
macro iterAround(ex)
    return esc(quote
        for xAdd in -r:r
            x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + xAdd
            if (x > 0 && x <= mainArrSize[1])
                for yAdd in -r:r
                    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + yAdd
                    if (y > 0 && y <= mainArrSize[2])
                        for zAdd in -r:r
                            z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + zAdd
                            if (z > 0 && z <= mainArrSize[3])
                                if ((abs(xAdd) + abs(yAdd) + abs(zAdd)) <= r)
                                    $ex
                                end
                            end
                        end
                    end
                end
            end
        end
    end)
end



"""
image - main image here computer tomography image
mainArrSize - dimensions of image
output - 4 dimensional array where first 3 dimaensions are the same as in case of the original image 
    and last dimension is of the same length as number of calculated features
r - size of the evaluated patch - the radius around each voxel that will be used to calculate local statistics

features and their location in 4th dimension
    1) original image
    2) means
    3) variances

"""
function calculateFeaturesExec(image, mainArrSize, output, r::Int, featuresNumb::Int)
    summ = 0.0
    sumCentered = 0.0
    lenn = UInt8(0)
    #get mean
    # @iterAround begin 
    #     lenn=lenn+1
    #     summ+=image[x,y,z]    
    # end
    # summ=summ/lenn#now summ acts as a mean
    # #get standard deviation
    # @iterAround sumCentered+= ((image[x,y,z]-summ )^2)

    #saving output
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y()))
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z()))

    if (x > 0 && x <= mainArrSize[1] && y > 0 && y <= mainArrSize[2] && z > 0 && z <= mainArrSize[3])
        output[x, y, z, 1, 1] = image[x, y, z] #original image
        output[x, y, z, 2, 1] = summ #mean
        output[x, y, z, 3, 1] = sumCentered #variance
    end#if

    return
end#calculateFeatures






"""
call function with out variable initialization
"""
function call_calculateFeatures(image, mainArrSize, r, featuresNumb, threads_CalculateFeatures, blocks_CalculateFeatures)
    output = CUDA.zeros(mainArrSize[1], mainArrSize[2], mainArrSize[3], (featuresNumb + 1), 1)
    @cuda threads = threads_CalculateFeatures blocks = blocks_CalculateFeatures calculateFeaturesExec(image, mainArrSize, output, r, featuresNumb)
    return output
end


"""
Enzyme definitions
"""
function calculateFeatures_Deff(image, d_image, mainArrSize, output, d_output, r, featuresNumb)

    Enzyme.autodiff_deferred(calculateFeaturesExec, Const, Duplicated(image, d_image), Const(mainArrSize), Duplicated(output, d_output), Const(r), Const(featuresNumb))
    return nothing
end




# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_calculateFeatures), image, mainArrSize, r, featuresNumb, threads_CalculateFeatures, blocks_CalculateFeatures)

    output = call_calculateFeatures(image, mainArrSize, r, featuresNumb, threads_CalculateFeatures, blocks_CalculateFeatures)

    function call_calculateFeatures_pullback(d_out_prim)
        # Allocate shadow memory.
        d_image = CUDA.ones(size(image))
        d_output = CuArray(collect(d_out_prim))
        @cuda threads = threads_CalculateFeatures blocks = blocks_CalculateFeatures calculateFeatures_Deff(image, d_image, mainArrSize, output, d_output, r, featuresNumb)
        f̄ = NoTangent()
        return f̄, d_image, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return output, call_calculateFeatures_pullback

end

#### LUX definietions
struct CalculateFeatures_str <: Lux.AbstractExplicitLayer
    r::Int
    featuresNumb::Int
    mainArrSize
    threads_CalculateFeatures::Tuple{Int64,Int64,Int64}
    blocks_CalculateFeatures::Tuple{Int64,Int64,Int64}
end

function calculateFeatures(r::Int, featuresNumb::Int, mainArrSize, threads_CalculateFeatures, blocks_CalculateFeatures)
    return CalculateFeatures_str(r, featuresNumb, mainArrSize, threads_CalculateFeatures, blocks_CalculateFeatures)
end

#no parameters for now
Lux.initialparameters(rng::AbstractRNG, l::CalculateFeatures_str) = NamedTuple()

function Lux.initialstates(::AbstractRNG, l::CalculateFeatures_str)::NamedTuple
    return (r=l.r, threads_CalculateFeatures=l.threads_CalculateFeatures, blocks_CalculateFeatures=l.blocks_CalculateFeatures, featuresNumb=l.featuresNumb)
end



"""
defining what the layer does when called
"""
function (l::CalculateFeatures_str)(origArr, ps, st::NamedTuple)
    return call_calculateFeatures(origArr, st.mainArrSize, st.r, st.featuresNumb, st.threads_CalculateFeatures, st.blocks_CalculateFeatures), st
end




# Nx, Ny, Nz = 32, 32, 32
# mainArrSize=(Nx,Ny,Nz)
# threads_CalculateFeatures=(8,4,8)
# blocks_CalculateFeatures = (cld(mainArrSize[1],threads_CalculateFeatures[1]), cld(mainArrSize[2],threads_CalculateFeatures[2])  , cld(mainArrSize[3],threads_CalculateFeatures[3]))
# image=CUDA.rand(mainArrSize[1],mainArrSize[2],mainArrSize[3])
# featuresNumb=2
# r=3

# output = call_calculateFeatures(image,mainArrSize,r,featuresNumb  ,threads_CalculateFeatures,blocks_CalculateFeatures )
# size(output)


# ress=Zygote.jacobian(call_calculateFeatures,image,mainArrSize,r,featuresNumb  ,threads_CalculateFeatures,blocks_CalculateFeatures)
# typeof(ress)
# maximum(ress[1])
# maximum(ress[2])


# """
# given image array calculates set of per voxel features 
# for the beginig it will calculate only mean of the surrounding area and standard deviation of surrounding area
# """
# function getPerVoxelFeatures(arr)



# end#    getPerVoxelFeatures


# Nx, Ny, Nz = 32, 32, 32
# oneSidePad = 1
# totalPad = oneSidePad*2
# dim_x,dim_y,dim_z= Nx+totalPad, Ny+totalPad, Nz+totalPad
# featureNumb=3
# conv1 = (in, out) -> Lux.Conv((3,3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())
# rng = Random.default_rng()

# modelConv=Lux.Chain(conv1(1,4),conv1(4,16),conv1(16,4),conv1(4,1))
# ps, st = Lux.setup(rng, modelConv)
# x = randn(rng, Float32, dim_x,dim_y,dim_z,featureNumb)
# x =reshape(x, (dim_x,dim_y,dim_z,featureNumb1,1))
# y_pred, st =Lux.apply(modelConv, x, ps, st) 
# size(y_pred)
