"""
Calculates weather the variance of the features is related to variance of the supervoxel probability
basically if features are relatively constant so some are looks mostly the same the probability
    of supervoxel presence shuld be relatively constant - hovewer spots where variance of features is high it indicates the boundry between 
    supervoxels
"""
using ChainRulesCore,Zygote,CUDA,Enzyme
using Zygote, Lux,CUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

#### test data


"""
stencil operation to calculate local mean
"""
@inline function getLocalMean(fullArr,Nx,Ny,Nz,x,y,z)
    res=0.0
    res+=fullArr[x,y,z,1,1]
    if x<(Nx+1) res+=fullArr[x+1,y,z,1,1]
    if x>1 res+=fullArr[x-1,y,z,1,1]
    if y<(Ny+1) res+=fullArr[x,y+1,z,1,1]
    if y>1 res+=fullArr[x,y-1,z,1,1]
    if z<(Nz+1) res+=fullArr[x,y,z+1,1,1]
    if z>1 res+=fullArr[x,y,z-1,1,1]
    return res/7
end

"""
stencil operation to calculate local variance
"""
@inline function getLocalvar(fullArr,Nx,Ny,Nz,x,y,z, mean)
    res=0.0
    res+=(fullArr[x,y,z,1,1]-mean)^2
    if x<(Nx+1) res+=(fullArr[x+1,y,z,1,1]-mean)^2
    if x>1 res+=(fullArr[x-1,y,z,1,1]-mean)^2
    if y<(Ny+1) res+=(fullArr[x,y+1,z,1,1]-mean)^2
    if y>1 res+=(fullArr[x,y-1,z,1,1]-mean)^2
    if z<(Nz+1) res+=(fullArr[x,y,z+1,1,1]-mean)^2
    if z>1 res+=(fullArr[x,y,z-1,1,1]-mean)^2
    return res/7
end


"""
given concatenated array with original array and variance of features and calculated probabilities of 
presence of given supervoxels  generally variance of supervoxel features and variance of probability should 
be strongly correlated and this correlation we will measure using logit cross entropy
in kernel we need to evaluate local variance of p 
probArr - the array with probabilities for given array and 
"""
function featureLoss_kern_(probArr, Aout,Nx,Ny,Nz)
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z()))
    #first we need to calculate variance of p
    # additionally we multiply by local value to reduce the importance of low probability regions
    Aout[x,y,z]=getLocalvar(probArr,Nx,Ny,Nz,x,y,z, getLocalMean(fullArr,Nx,Ny,Nz,x,y,z))*probArr[x,y,z,1,1]
  
    return nothing
end

function featureLoss_kern__Deff(fullArr
    , d_fullArr, Aout
    , dAout,Nx,Ny,Nz)
    Enzyme.autodiff_deferred(featureLoss_kern_, Const,
     Duplicated(fullArr, d_fullArr)
     , Duplicated(Aout, dAout)
     ,Const(Nx)
     ,Const(Ny)
     ,Const(Nz)
    )
    return nothing
end

function call_featureLoss_kern_(A,probArr,Nx,Ny,Nz,threads_featureLoss_kern_,blocks_featureLoss_kern_)
    Aout = CUDA.zeros(Nx, Ny, Nz,1,1 ) 
    @cuda threads = threads_featureLoss_kern_ blocks = blocks_featureLoss_kern_ featureLoss_kern_(probArr,  Aout,Nx,Ny,Nz)
    return Aout
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_featureLoss_kern_), A, probArr,Nx,Ny,Nz,threads_featureLoss_kern_,blocks_featureLoss_kern_)
    
    Aout = call_featureLoss_kern_( probArr,Nx,Ny,Nz,threads_featureLoss_kern_,blocks_featureLoss_kern_)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)
        dp = CUDA.ones(size(probArr))
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)
        @cuda threads = threads_featureLoss_kern_ blocks = blocks_featureLoss_kern_ featureLoss_kern__Deff(probArr, dp, Aout, CuArray(collect(dAout)),Nx,Ny,Nz,threads_featureLoss_kern_,blocks_featureLoss_kern_)
       
        return NoTangent(), dp,NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent()
    end   
    return Aout, call_test_kernel1_pullback

end

#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct featureLoss_kern__str<: Lux.AbstractExplicitLayer
    Nx::Int
    Ny::Int
    Nz::Int
    featureNumb::Int
    threads_featureLoss_kern_::Tuple{Int,Int,Int}
    blocks_featureLoss_kern_::Tuple{Int,Int,Int}
end

function featureLoss_kern__layer(Nx,Ny,Nz,threads_featureLoss_kern_,blocks_featureLoss_kern_,featureNumb)
    return featureLoss_kern__str(Nx,Ny,Nz,threads_featureLoss_kern_,blocks_featureLoss_kern_,featureNumb)
end

Lux.initialparameters(rng::AbstractRNG, l::featureLoss_kern__str)=NamedTuple()
"""
https://stackoverflow.com/questions/52035775/in-julia-1-0-how-to-set-a-named-tuple-with-only-one-key-value-pair
in order to get named tuple with single element put comma after
"""
function Lux.initialstates(::AbstractRNG, l::featureLoss_kern__str)::NamedTuple
    return (Nx=l.Nx,Ny=l.Ny,Nz=l.Nz,threads_featureLoss_kern_=l.threads_featureLoss_kern_
    ,blocks_featureLoss_kern_=l.blocks_featureLoss_kern_,featureNumb=l.featureNumb )
end

function (l::featureLoss_kern__str)(x, ps, st::NamedTuple)
    #below we just get the local variance of probabilities now we need to compare it with features variance
    calculated_variances=call_featureLoss_kern_(x[1][:,:,:,1,:],ps.p,l.Nx,l.Ny,l.Nz,l.threads_featureLoss_kern_,l.blocks_featureLoss_kern_ )
    #we initialize with the value from spread loss
    feature_lossTotal=x[2]
    #we iterate over all features and compares them 
    for featurIndex in l:featureNumb
        feature_lossTotal+=Flux.binary_focal_loss(x[:,:,:,2+featurIndex,:],calculated_variances)
    end for

    return (x,feature_lossTotal) ,st
end
