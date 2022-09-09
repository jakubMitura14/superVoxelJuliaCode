""" 
very simple layer that just perform elementwise multiplication
"""

using Revise

using Zygote, Lux,Enzyme,Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote



struct MultLayer_str<: Lux.AbstractExplicitLayer
    dims::Tuple{Int64, Int64, Int64, Int64, Int64}
end

function MultLayer(dims::Tuple{Int64, Int64, Int64, Int64, Int64})::MultLayer_str
    return MultLayer_str(dims)
end

function Lux.initialparameters(rng::AbstractRNG, l::MultLayer_str)
    return (p=rand(rng, Float32, l.dims...),)
end

Lux.initialstates(::AbstractRNG, l::MultLayer_str)=NamedTuple()



function (l::MultLayer_str)(origArr, ps, st::NamedTuple)
      return (origArr.*ps.p),st
end