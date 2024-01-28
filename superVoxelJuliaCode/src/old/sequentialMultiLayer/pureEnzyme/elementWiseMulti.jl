
import NNlib, Optimisers, Plots, Random, Statistics,  Zygote, Lux,CUDA


"""
get element wise multiplication - in order to simplify we will treat it as a one dimensional array
source - the array that get multiplied
toMultiplyWith - the array that we will learn - basically parameters
Aout - for storing output
Nx - length (when treated array as single dimension) of the whole array ( source and toMultiplyWith must have the same shape)
"""
function elementWise_kernel(source,toMultiplyWith,Aout,Nx)
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    #check if we are in range
    if(x<Nx) 
        Aout[x]= source[x]*toMultiplyWith[x]
    end    
    return nothing
end

function elementWise_Deff(source, d_source,
    toMultiplyWith,d_toMultiplyWith,
    Aout, d_Aout,    Nx
    )
    Enzyme.autodiff_deferred(elementWise_kernel, Const,
     Duplicated(source, d_source)
     Duplicated(toMultiplyWith,d_toMultiplyWith)
     , Duplicated(Aout, dAout)
    )
    return nothing
end
