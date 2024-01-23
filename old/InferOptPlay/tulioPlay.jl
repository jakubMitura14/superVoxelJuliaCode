using  LoopVectorization
using Tullio
using OffsetArrays
using TensorOperations

using Pkg; Pkg.add("Tullio")
using Tullio, Test
M = ones(3, 7)

@tullio S[1,c] := M[r,c]  # sum over r ∈ 1:3, for each c ∈ 1:7


using Tullio, OffsetArrays

# A convolution with cyclic indices
mat = zeros(10,10,10); mat[2,2,2] = 101; mat[10,10,4] = 1;
offsets = [(a,b) for a in -2:2 for b in -2:2 if a>=b] # vector of tuples
out = ones(10,10,10)
@tullio out[x,y,z] = begin
        a,b = offsets[k]
        i = clamp(x+a, extrema(axes(mat,1))...)
        j = clamp(y+b, extrema(axes(mat,2))...) # can be written clamp(y+b)
        @inbounds mat[i, j, z] * 10
    end # ranges of x,y read from out[x,y,1]



sum(out)




    @tullio out[x,y,c] := begin
        xi = mod(x+i, axes(mat,1)) # xi = ... means that it won't be summed,
        yj = mod(y+j, axes(mat,2))
        @inbounds trunc(Int, mat[xi, yj, c] * kern[i,j]) # and disables automatic @inbounds,
    end (x in 1:10, y in 1:10) # and prevents range of x from being inferred.
    






















using Tullio, OffsetArrays

# A convolution with cyclic indices
mat = zeros(10,10,1); mat[2,2] = 101; mat[10,10] = 1;
@tullio kern[i,j] := 1/(1+i^2+j^2)  (i in -3:3, j in -3:3)

@tullio out[x,y,c] := begin
    xi = mod(x+i, axes(mat,1)) # xi = ... means that it won't be summed,
    yj = mod(y+j, axes(mat,2))
    @inbounds trunc(Int, mat[xi, yj, c] * kern[i,j]) # and disables automatic @inbounds,
end (x in 1:10, y in 1:10) # and prevents range of x from being inferred.

# A stencil?
offsets = [(a,b) for a in -2:2 for b in -2:2 if a>=b] # vector of tuples

@tullio out[x,y,1] = begin
        a,b = offsets[k]
        i = clamp(x+a, extrema(axes(mat,1))...)
        j = clamp(y+b, extrema(axes(mat,2))...) # can be written clamp(y+b)
        @inbounds mat[i, j, 1] * 10
    end # ranges of x,y read from out[x,y,1]
