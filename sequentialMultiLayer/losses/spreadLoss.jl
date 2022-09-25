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
we will calculate the distance of any given point relative to 3 corners and multiply
by value of p - hope is we will get workable triangulation and reducing the variance will 
lead to reduction of cluster spread
we will save the data in 3 separate channels of Aout
choice of corners is arbitrary
"""
function spreadKern(p, Aout,Nx,Ny,Nz)
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z()))
    #check if we are in range
    if(x<Nx && y<Ny && z<Nz) 
        #dist to top left posterior corner
        Aout[x, y, z,1]= (((x-1)^2)+((y-1)^2) +((z-1)^2))*(p[x,y,z])^2
        #dist to bottom right anterior corner
        Aout[x, y, z,2]= (((x-Nx)^2)+((y-Ny)^2) +((z-Nz)^2))*(p[x,y,z])^2
        #dist to top right anterior corner
        Aout[x, y, z,3]= (((x-1)^2)+((y-Ny)^2) +((z-Nz)^2))*(p[x,y,z])^2       
    end    
    return nothing
end

function spreadKern_Deff( p
    , dp, Aout
    , dAout,Nx,Ny,Nz)
    Enzyme.autodiff_deferred(spreadKern, Const,
     Duplicated(p, dp)
     , Duplicated(Aout, dAout)
     ,Const(Nx)
     ,Const(Ny)
     ,Const(Nz)
    )
    return nothing
end

function call_spreadKern(A, p,Nx,Ny,Nz,threads_spreadKern,blocks_spreadKern)
    Aout = CUDA.zeros(Nx, Ny, Nz,3,1 ) 
    @cuda threads = threads_spreadKern blocks = blocks_spreadKern spreadKern(p,  Aout,Nx,Ny,Nz)
    return Aout
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_spreadKern), A, p,Nx,Ny,Nz,threads_spreadKern,blocks_spreadKern)
    
    Aout = call_spreadKern( p,Nx,Ny,Nz,threads_spreadKern,blocks_spreadKern)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)
        dp = CUDA.ones(size(p))
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)
        @cuda threads = threads_spreadKern blocks = blocks_spreadKern spreadKern_Deff(p, dp, Aout, CuArray(collect(dAout)),Nx,Ny,Nz,threads_spreadKern,blocks_spreadKern)
       
        return NoTangent(), dp,NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent()
    end   
    return Aout, call_test_kernel1_pullback

end

#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct spreadKern_str<: Lux.AbstractExplicitLayer
    Nx::Int
    Ny::Int
    Nz::Int
    probMapChannel::Int
    featuresStartChannel::Int
    threads_spreadKern::Tuple{Int,Int,Int}
    blocks_spreadKern::Tuple{Int,Int,Int}
end

function spreadKern_layer(Nx,Ny,Nz,probMapChannel,featuresStartChannel,threads_spreadKern,blocks_spreadKern)
    return spreadKern_str(Nx,Ny,Nz,probMapChannel,featuresStartChannel,threads_spreadKern,blocks_spreadKern)
end

Lux.initialparameters(rng::AbstractRNG, l::spreadKern_str)=NamedTuple()

function Lux.initialstates(::AbstractRNG, l::spreadKern_str)::NamedTuple
    return (Nx=l.Nx,Ny=l.Ny,Nz=l.Nz,probMapChannel=l.probMapChannel,featuresStartChannel=l.featuresStartChannel,
    threads_spreadKern=l.threads_spreadKern,blocks_spreadKern=l.blocks_spreadKern )
end

function (l::spreadKern_str)(x, ps, st::NamedTuple)
    calculated=call_spreadKern(x[1][:,:,:,l.probMapChannel,:],ps.p,l.Nx,l.Ny,l.Nz,l.threads_spreadKern,l.blocks_spreadKern )
    # so using triangulation we add variance in position relative to 3 corners x[2] is the loss that is already accumulated from previous supervoxels
    spreadLoss=var(calculated[:,:,:,1,1])+var(calculated[:,:,:,2,1])+var(calculated[:,:,:,3,1])+x[2]
     
     return (x[1],spreadLoss) ,st
end


# function loss_function_spread(model, ps, st, x)
#     y_pred, st = Lux.apply(model, x, ps, st)
#     res= var(y_pred[:,:,:,1,1])+var(y_pred[:,:,:,2,1])+var(y_pred[:,:,:,3,1])
#     return res, st, ()
# end


# Nx, Ny, Nz = 16, 16, 16
# oneSidePad = 1

# threads_spreadKern = (4, 4, 4)
# blocks_spreadKern = (4, 4, 4)
# rng = Random.default_rng()


# l = spreadKern_layer(Nx,Ny,Nz,oneSidePad,threads_spreadKern,blocks_spreadKern)
# ps, st = Lux.setup(rng, l)
# println("Parameter Length: ", Lux.parameterlength(l), "; State Length: ",
#         Lux.statelength(l))

# x = randn(rng, Float32, Nx, Ny,Nz,1,1)
# x= CuArray(x)
# # testing weather forward pass runs
# y_pred, st =Lux.apply(l, x, ps, st)



# model = Lux.Chain(KernelA(Nx),KernelA(Nx)) 
# opt = Optimisers.Adam(0.003)



# tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
# vjp_rule = Lux.Training.ZygoteVJP()


# function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data,
#     epochs::Int)
#    # data = data .|> Lux.gpu
#     for epoch in 1:epochs
#         grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
#                                                                 data, tstate)
#         @info epoch=epoch loss=loss
#         tstate = Lux.Training.apply_gradients(tstate, grads)
#     end
#     return tstate
# end
# # one epoch just to check if it runs
# tstate = main(tstate, vjp_rule, x,1)
# #training 
# tstate = main(tstate, vjp_rule, x,1000)

