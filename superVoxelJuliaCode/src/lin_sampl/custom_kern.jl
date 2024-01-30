using Pkg
using ChainRulesCore,Zygote,CUDA,Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux,LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays
using LinearAlgebra
using Images,ImageFiltering
using Revise


# @device_override Base.ceil(x::Float64) = ccall("extern __nv_ceil", llvmcall, Cdouble, (Cdouble,), x)
# @device_override Base.ceil(x::Float32) = ccall("extern __nv_ceilf", llvmcall, Cfloat, (Cfloat,), x)
# @device_override Base.floor(f::Float64) = ccall("extern __nv_floor", llvmcall, Cdouble, (Cdouble,), f)
# @device_override Base.floor(f::Float32) = ccall("extern __nv_floorf", llvmcall, Cfloat, (Cfloat,), f)


function sigmoid(x::Float32)::Float32
    return 1 / (1 + exp(-x))
end

# function @my_ceil(x::Float32)::Float32
#     # return x+(1-(x%1))
#     return round(x+0.5)
# end
# function @my_floor(x::Float32)::Float32
#     # return x+(1-(x%1))
#     return round(x-0.5)
# end

macro my_ceil(x)
  return  esc(quote
  round($x+0.5)
end)
end

macro my_floor(x)
    return  esc(quote
    round($x-0.5)
  end)
  end

# @@my_ceil(1.99)

# @my_ceil =(x)->round(x+0.5)
# @my_floor =(x)->round(x-0.5)


# coord_x = rand()*100
# coord_y = rand()*100
# coord_z = rand()*100

# sum_dist(coord_x, coord_y, coord_z)


# sum_dist(1.1,1.1,1.9)
# sum_dist(1.1,1.1,1.1)
# sum_dist(1.1,1.9,1.1)
# sum_dist(1.9,1.1,1.1)
# sum_dist(1.9,1.1,1.9)
# sum_dist(1.7,1.1,1.9)



"""
simple kernel friendly interpolator - given float coordinates and source array will 
1) look for closest integers in all directions and calculate the euclidean distance to it 
2) calculate the weights for each of the 8 points in the cube around the pointadding more weight the closer the point is to integer coordinate
"""

function ThreeDLinInterpol(coord_x,coord_y,coord_z,source_arr)
    ## first we get the total distances of all points to be able to normalize it later
    dist_sum=0.0    
    dist_sum+=sqrt((coord_x-@my_floor(coord_x))^2+(coord_y-@my_floor(coord_y))^2+(coord_z-@my_floor(coord_z))^2)
    dist_sum+=sqrt((coord_x-@my_ceil(coord_x))^2+(coord_y-@my_floor(coord_y))^2+(coord_z-@my_floor(coord_z))^2)
    dist_sum+=sqrt((coord_x-@my_floor(coord_x))^2+(coord_y-@my_ceil(coord_y))^2+(coord_z-@my_floor(coord_z))^2)
    dist_sum+=sqrt((coord_x-@my_floor(coord_x))^2+(coord_y-@my_floor(coord_y))^2+(coord_z-@my_ceil(coord_z))^2)
    dist_sum+=sqrt((coord_x-@my_ceil(coord_x))^2+(coord_y-@my_ceil(coord_y))^2+(coord_z-@my_floor(coord_z))^2)
    dist_sum+=sqrt((coord_x-@my_floor(coord_x))^2+(coord_y-@my_ceil(coord_y))^2+(coord_z-@my_ceil(coord_z))^2)
    dist_sum+=sqrt((coord_x-@my_ceil(coord_x))^2+(coord_y-@my_floor(coord_y))^2+(coord_z-@my_ceil(coord_z))^2)
    dist_sum+=sqrt((coord_x-@my_ceil(coord_x))^2+(coord_y-@my_ceil(coord_y))^2+(coord_z-@my_ceil(coord_z))^2)
    # ## now we get the final value by weightes summation
    res= source_arr[Int(@my_floor(coord_x)),Int(@my_floor(coord_y)),Int(@my_floor(coord_z))]*(sqrt((coord_x-@my_floor(coord_x))^2+(coord_y-@my_floor(coord_y))^2+(coord_z-@my_floor(coord_z))^2)/dist_sum)
    res+= source_arr[Int(@my_ceil(coord_x)) ,Int(@my_floor(coord_y)),Int(@my_floor(coord_z))]*(sqrt((coord_x-@my_ceil(coord_x))^2+(coord_y-@my_floor(coord_y))^2+(coord_z-@my_floor(coord_z))^2)/dist_sum)
    res+= source_arr[Int(@my_floor(coord_x)),Int(@my_ceil(coord_y)) ,Int(@my_floor(coord_z))]*(sqrt((coord_x-@my_floor(coord_x))^2+(coord_y-@my_ceil(coord_y))^2+(coord_z-@my_floor(coord_z))^2)/dist_sum)
    res+= source_arr[Int(@my_floor(coord_x)),Int(@my_floor(coord_y)),Int(@my_ceil(coord_z))]*(sqrt((coord_x-@my_floor(coord_x))^2+(coord_y-@my_floor(coord_y))^2+(coord_z-@my_ceil(coord_z))^2)/dist_sum)
    res+= source_arr[Int(@my_ceil(coord_x)) ,Int(@my_ceil(coord_y)) ,Int(@my_floor(coord_z))]*(sqrt((coord_x-@my_ceil(coord_x))^2+(coord_y-@my_ceil(coord_y))^2+(coord_z-@my_floor(coord_z))^2)/dist_sum)
    res+= source_arr[Int(@my_floor(coord_x)),Int(@my_ceil(coord_y)) ,Int(@my_ceil(coord_z))]*(sqrt((coord_x-@my_floor(coord_x))^2+(coord_y-@my_ceil(coord_y))^2+(coord_z-@my_ceil(coord_z))^2)/dist_sum)
    res+= source_arr[Int(@my_ceil(coord_x)) ,Int(@my_floor(coord_y)),Int(@my_ceil(coord_z))]*(sqrt((coord_x-@my_ceil(coord_x))^2+(coord_y-@my_floor(coord_y))^2+(coord_z-@my_ceil(coord_z))^2)/dist_sum)
    res+= source_arr[Int(@my_ceil(coord_x))  ,Int(@my_ceil(coord_y)) ,Int(@my_ceil(coord_z))]*(sqrt((coord_x-@my_ceil(coord_x))^2+(coord_y-@my_ceil(coord_y))^2+(coord_z-@my_ceil(coord_z))^2)/dist_sum)
    
    return res
end

#### main kernel
function testKern(prim_A,A, p, Aout,Nx)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) 

    x_p=A[x,y,z,1,1]*(Nx-3)+2 
    y_p =A[x,y,z,2,1]*(Nx-3)+2 
    z_p =A[x,y,z,3,1]*(Nx-3)+2 

    diff=1.12
    res=ThreeDLinInterpol(x_p-diff,y_p,z_p,prim_A)
    res-=ThreeDLinInterpol(x_p+diff,y_p,z_p,prim_A)
    
    res+=ThreeDLinInterpol(x_p,y_p-diff,z_p,prim_A)
    res-=ThreeDLinInterpol(x_p,y_p+diff,z_p,prim_A)
    
    
    res+=ThreeDLinInterpol(x_p,y_p,z_p-diff,prim_A)
    res-=ThreeDLinInterpol(x_p,y_p,z_p+diff,prim_A)


    # res=prim_A[x_p-1,y_p,z_p] - prim_A[x_p+1,y_p,z_p] + prim_A[x_p,y_p-1,z_p] - prim_A[x_p,y_p+1,z_p] + prim_A[x_p,y_p,z_p-1] - prim_A[x_p,y_p,z_p+1]
    # Aout[x*4+y*2+z]=res
    Aout[(x-1)*4+(y-1)*2+z]=res
    
    return nothing
end#testKern

# """
# iterate over array that is treated as one dimensional with given length lengthh as argument
# amount of iterations needed is also passed as an argument - iterLoop
# """
# macro interpolate_on_axis(iterLoop,lengthh, ex)
#   return  esc(quote
#   i = UInt32(0)
#   @unroll for j in 0:($iterLoop)
#     i= threadIdxX()+(threadIdxY()-1)*blockDimX()+ j* blockDimX()*blockDimY()
#     if(i<=$lengthh) 
#       $ex
#     end 
#     end 
# end)
