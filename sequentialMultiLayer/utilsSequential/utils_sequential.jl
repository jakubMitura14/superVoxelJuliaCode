
"""
given two inputs return given entry
"""


using Revise,Logging
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
    return(to_select=l.to_select, )
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


"""
get 2 entries and create tuple from them
"""
function myGetTuple(a,b)
   return (a,b)
end  


############## for printing shapes



struct Print_debug_str<: Lux.AbstractExplicitLayer
    name
end

function Print_debug(name)::Print_debug_str
    return Print_debug_str(name)
end

Lux.initialparameters(rng::AbstractRNG, l::Print_debug_str)=NamedTuple()

function Lux.initialstates(::AbstractRNG, l::Print_debug_str)
    return(name=l.name, )
end #initialstates    



flat(arr::Tuple) = mapreduce(x -> isa(x, Tuple) ? flat(x) : x, append!, arr,init=[])

function (l::Print_debug_str)(x, ps, st::NamedTuple)
    @info " nam $(l.name) type  $(typeof(x)) "
    flattened= flat((x,))
    indexx=0
    for entry in flattened
        indexx+=1
        if(isa(x, Array))
            @info " indexx $(indexx) size  $(size(entry)) "
        end#if isa    
    end#for    

    return x,st
end