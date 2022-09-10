"""
here we need to compare the last supervoxel map with accumulated previous ones
main idea is that we want to achieve maximum disagreement in places where one probMap gives high value
other should give small - places where both array have small values are ok 
"""
"""
we can calculate the variance of current position relative to 3
    corners (triangulation)
    so we get euclidean distance to corner a times p we save then the same relative to corner b and C
    finally from saved 3 arrays we calculate variances and add them up 
"""

using ChainRulesCore,Zygote,CUDA,Enzyme
using Zygote, Lux,CUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote

#### test data


"""
given probability map of previous supervoxels and current one we compare them and hope we will
have values as diffrent as possible 
"""
function disagreeKern(previous_prob_maps,current_probMap, Aout,Nx,Ny,Nz)
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z()))
    #check if we are in range
    if(x<Nx && y<Ny && z<Nz) 
        #checking diffrence squared and multiply by multiplied values - as we care basically only about places where we have hihgh on high
        Aout[x, y, z,1,1]= ((previous_prob_maps[x, y, z,1,1]-current_probMap[x, y, z,1,1])^2)*previous_prob_maps[x, y, z,1,1]*current_probMap[x, y, z,1,1]
    end    
    return nothing
end

function disagreeKern_Deff( previous_prob_maps,d_previous_prob_maps
    ,current_probMap,d_current_probMap, Aout
    , dAout,Nx,Ny,Nz)
    Enzyme.autodiff_deferred(disagreeKern, Const,
     Duplicated(previous_prob_maps,d_previous_prob_maps)
     Duplicated(current_probMap,d_current_probMap)
     , Duplicated(Aout, dAout)
     ,Const(Nx)
     ,Const(Ny)
     ,Const(Nz)
    )
    return nothing
end

function call_disagreeKern(current_probMap,d_current_probMap,Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)
    Aout = CUDA.zeros(Nx, Ny, Nz,1,1 ) 
    @cuda threads = threads_disagreeKern blocks = blocks_disagreeKern disagreeKern(current_probMap,d_current_probMap,Aout,Nx,Ny,Nz)
    return Aout
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_disagreeKern), A, p,Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)
    
    Aout = call_disagreeKern( p,Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)
        dp = CUDA.ones(size(p))
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)
        @cuda threads = threads_disagreeKern blocks = blocks_disagreeKern disagreeKern_Deff(p, dp, Aout, CuArray(collect(dAout)),Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)
       
        return NoTangent(), dp,NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent()
    end   
    return Aout, call_test_kernel1_pullback

end

#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct disagreeKern_str<: Lux.AbstractExplicitLayer
    Nx::Int
    Ny::Int
    Nz::Int
    probMapChannel::Int
    featuresStartChannel::Int
    threads_disagreeKern::Tuple{Int,Int,Int}
    blocks_disagreeKern::Tuple{Int,Int,Int}
end

function disagreeKern_layer(Nx,Ny,Nz,,probMapChannel,featuresStartChannel,threads_disagreeKern,blocks_disagreeKern)
    return disagreeKern_str(Nx,Ny,Nz,,probMapChannel,featuresStartChannel,threads_disagreeKern,blocks_disagreeKern)
end

Lux.initialparameters(rng::AbstractRNG, l::disagreeKern_str)=NamedTuple()

function Lux.initialstates(::AbstractRNG, l::disagreeKern_str)::NamedTuple
    return (Nx=l.Nx,Ny=l.Ny,Nz=l.Nz,probMapChannel=l.probMapChannel,featuresStartChannel=l.featuresStartChannel,
    threads_disagreeKern=l.threads_disagreeKern,blocks_disagreeKern=l.blocks_disagreeKern )
end

function (l::disagreeKern_str)(x, ps, st::NamedTuple)
    calculated=call_disagreeKern(x[1][:,:,:,l.probMapChannel,:],ps.p,l.Nx,l.Ny,l.Nz,l.threads_disagreeKern,l.blocks_disagreeKern )
    # so using triangulation we add variance in position relative to 3 corners x[2] is the loss that is already accumulated from previous supervoxels
    spreadLoss=var(calculated[:,:,:,1,1])+var(calculated[:,:,:,2,1])+var(calculated[:,:,:,3,1])+x[2]
     
     return (x[1],spreadLoss) ,st
end

