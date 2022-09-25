"""
here we need to compare the last supervoxel map with accumulated previous ones
main idea is that we want to achieve maximum disagreement in places where one probMap gives high value
other should give small - places where both array have small values are ok 

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
        Aout[x, y, z,2,1]= ((previous_prob_maps[x, y, z,1,1]+current_probMap[x, y, z,1,1]))#for futher iterations

    end    
    return nothing
end

function disagreeKern_Deff( previous_prob_maps,d_previous_prob_maps
    ,current_probMap,d_current_probMap, Aout
    , dAout,Nx,Ny,Nz)
    Enzyme.autodiff_deferred(disagreeKern, Const,
     Duplicated(previous_prob_maps,d_previous_prob_maps)
     ,Duplicated(current_probMap,d_current_probMap)
     , Duplicated(Aout, dAout)
     ,Const(Nx)
     ,Const(Ny)
     ,Const(Nz)
    )
    return nothing
end

function call_disagreeKern(previous_prob_maps,current_probMap,d_current_probMap,Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)
    Aout = CUDA.zeros(Nx, Ny, Nz,2,1 ) 
    @cuda threads = threads_disagreeKern blocks = blocks_disagreeKern disagreeKern(previous_prob_maps,current_probMap,Aout,Nx,Ny,Nz)
    return Aout
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_disagreeKern), previous_prob_maps,current_probMap,Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)
    
    Aout = call_disagreeKern( previous_prob_maps,current_probMap,Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function disagreeKern_pullback(dAout)

        d_previous_prob_maps= CUDA.ones(size(previous_prob_maps))
        d_current_probMap= CUDA.ones(size(current_probMap))
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)
        @cuda threads = threads_disagreeKern blocks = blocks_disagreeKern disagreeKern_Deff(
            previous_prob_maps,d_previous_prob_maps,current_probMap,d_current_probMap
            ,Aout,CuArray(collect(dAout)),Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)
       
        return NoTangent(), d_previous_prob_maps,d_current_probMap ,NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent()
    end   
    return Aout, disagreeKern_pullback

end

#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct disagreeKern_str<: Lux.AbstractExplicitLayer
    Nx::Int
    Ny::Int
    Nz::Int
    threads_disagreeKern::Tuple{Int,Int,Int}
    blocks_disagreeKern::Tuple{Int,Int,Int}
end

function disagreeKern_layer(Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)
    return disagreeKern_str(Nx,Ny,Nz,threads_disagreeKern,blocks_disagreeKern)
end

Lux.initialparameters(rng::AbstractRNG, l::disagreeKern_str)=NamedTuple()

function Lux.initialstates(::AbstractRNG, l::disagreeKern_str)::NamedTuple
    return (Nx=l.Nx,Ny=l.Ny,Nz=l.Nz,
    threads_disagreeKern=l.threads_disagreeKern,blocks_disagreeKern=l.blocks_disagreeKern )
end

function (l::disagreeKern_str)(x, ps, st::NamedTuple)
    channelsNum= size(x)[4]
    concated=call_disagreeKern(x[1][:,:,:,1,:],x[1][:,:,:,2,:],l.Nx,l.Ny,l.Nz,l.threads_disagreeKern,l.blocks_disagreeKern )
    #we just sum elementwise scaled diffrence
    lossVal=sum(concated[:,:,:,1,:])
    return (myCatt4(concated[:,:,:,2,:], x[:,:,:,3:channelsNum,:] ) ,lossVal) ,st
end

