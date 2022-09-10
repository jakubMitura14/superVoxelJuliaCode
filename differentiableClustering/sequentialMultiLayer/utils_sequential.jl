
"""
given two inputs return given entry
"""
using Revise

using Zygote, Lux,Enzyme,Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote


struct SelectTupl_str<: Lux.AbstractExplicitLayer
    to_select::Int64
end

function SelectTupl(to_select::Int64)::SelectTupl_str
    return SelectTupl_str(to_select)
end

Lux.initialparameters(rng::AbstractRNG, l::SelectTupl_str)=NamedTuple()

function Lux.initialstates(::AbstractRNG, l::SelectTupl_str)
    return(to_select=l,to_select, )
end #initialstates    



function (l::SelectTupl_str)(origTupl, ps, st::NamedTuple)
      return origTupl[l.to_select],st
end


"""
concatenate on 4th (channel) dimension
"""
function myCatt4(a,b)
   return cat(a,b;dims=4)
end  