using CUDA, Enzyme, Test
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl
# and https://enzyme.mit.edu/julia/generated/box/


using CUDA
using Enzyme
using Test

"""
for some reason raising to the power greater then 2 give erro - hence this macro enable caling power of two multiple times
"""
macro myPowTwo(ex,num)
    exIn= quote (($ex)^2) end
    for i in 1:(num-1)
        exIn= quote (($exIn)^2) end
    end
    return esc(:( $exIn ))
end


"""
normalizes a number so the all numbers from supplied series sum to 1
"""
@inline function inlineNormalize(el::Float32, a::Float32,b::Float32,c::Float32)::Float32
    # return @myPowTwo(((1-((el)/(a+b+c+d+e+f)))+1),8)
    return @myPowTwo(((1-((el)/(a+b+c)))+1),8)
end#inlineNormalize

@inline function normPair(el::Float32,a::Float32,b::Float32)::Float32
    return @myPowTwo((el/(a+b))+1,4)
end#normPair

"""
raise to power in order to increase the diffrence between small and big number sth like max but not exact and differentiable
but keep it normalized
"""
@inline function pow_norm(el, a,b,c,d,e,f)::Float32
    return inlineNormalize(el, a,b,c,d,e,f)  /(inlineNormalize(a, a,b,c,d,e,f) 
            +inlineNormalize(b, a,b,c,d,e,f) 
            +inlineNormalize(c, a,b,c,d,e,f) 
            +inlineNormalize(d, a,b,c,d,e,f) +
            inlineNormalize(e, a,b,c,d,e,f) 
            +inlineNormalize(f, a,b,c,d,e,f) )
end#raiseToPow


"""
given 2 numbers return sth like max
"""
@inline  function alaMax(a,b)::Float32
   return ((normPair(a,a,b) /(normPair(a,a,b) + normPair(b,a,b) ))*a) + ((normPair(b,a,b) /(normPair(a,a,b) + normPair(b,a,b) ))*b)
end#alaMax

"""
process neighberhood of given voxel looks for smallest probability and
    in direction of smallest probability checking which number is bigger 
    current or this in direction of smallest prob
"""
@inline function processNeighbours(currA,currP, pa,pb,pc,pd
    ,pe,pf, aA, bA, cA, dA, eA, fA)::Float32
    return pow_norm(pa, pa,pb,pc,pd,pe,pf)*alaMax(currA,aA)+
    pow_norm(pb, pa,pb,pc,pd,pe,pf)*alaMax(currA,bA)+
    pow_norm(pc, pa,pb,pc,pd,pe,pf)*alaMax(currA,cA)+
    pow_norm(pd, pa,pb,pc,pd,pe,pf)*alaMax(currA,dA)+
    pow_norm(pe, pa,pb,pc,pd,pe,pf)*alaMax(currA,eA)+
    pow_norm(pf, pa,pb,pc,pd,pe,pf)*alaMax(currA,fA)
end#processNeighbours


@inline function myDiv(a::Float32,b::Float32)::Float32
        return @myPowTwo((Float32(a)/(Float32(a)+Float32(b)))+1,4)
end    


function mul_kernel(Nx::Int64,Ny::Int64,Nz::Int64,A::CuDeviceArray{Float32, 3}
    ,p::CuDeviceArray{Float32, 3},Aout::CuDeviceArray{Float32, 3})
    #adding one bewcouse of padding
    x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))+1
    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))+1
    z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))+1
    #Aout[x,y,z]= alaMax(p[x,y,z],p[x,y,z])
    #normPair(Float32(A[x,y,z]),Float32(A[x,y,z]),Float32(A[x+1,y,z]),Int32(20))  #

    Aout= p[x,y,z]*p[x,y,z]# myDiv(p[x,y,z],p[x-1,y,z])#*alaMax(A[x,y,z],A[x-1,y,z])
    
    # Aout[x,y,z]=processNeighbours(
    #     A[x,y,z]
    #     ,p[x,y,z]
    #     ,p[x-1,y,z]
    #     ,p[x+1,y,z]
    #     ,p[x,y-1,z]
    #     ,p[x,y+1,z]
    #     ,p[x,y,z-1]
    #     ,p[x,y,z+1]
    #     ,A[x-1,y,z]
    #     ,A[x+1,y,z]
    #     ,A[x,y-1,z]
    #     ,A[x,y+1,z]
    #     ,A[x,y,z-1]
    #     ,A[x,y,z+1])


    # if(p[x+1,y,z]< p[x-1,y,z])
    #     Aout[x,y,z]=A[x,y,z]*p[x,y,z]
    # end
 

    return nothing
end

function grad_mul_kernel(Nx,Ny,Nz,A, dA,p,dp,Aout,dAout)
    Enzyme.autodiff_deferred(mul_kernel, Const, Const(Nx), Const(Ny), Const(Nz)
    , Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout,dAout)    )
    return nothing
end


Nx,Ny,Nz= 64+2,64+2,64+2

A = CuArray(Float32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
dA = similar(A)


p = CuArray(Float32.(rand(0.0:1.0,Nx,Ny,Nz)))
dp = CUDA.ones(Nx,Ny,Nz)

Aout = CUDA.zeros(Nx,Ny,Nz)
dAout = CUDA.ones(Nx,Ny,Nz)

dA .= 1
@cuda threads= (8,8,8) blocks=(8,8,8) grad_mul_kernel(Nx,Ny,Nz,A, dA,p,dp,Aout,dAout)

maximum(dp)
minimum(dp)


maximum(Aout)

"""
simple cost function will be designed as just the number of unique values
"""










# currA=1101
# currP=0.2
# powwA=24
# powwP=50
# pa=0.1
# pb=0.8
# pc=0.4
# pd=0.6
# pe=0.9
# pf=0.2

# aA=10004
# bA=200
# cA=10
# dA=10
# eA=7000000
# fA=2134

# processNeighbours(currA,currP,powwA, powwP, pa,pb,pc,pd
#     ,pe,pf, aA, bA, cA, dA, eA, fA)
# #6700








# alaMax(1,15,16)

# a=0.2
# b=0.6
# c=0.8
# d=0.11
# e=0.3
# f=0.5

# el=d
# poww=20
# res = pow_norm(el,poww, a,b,c,d,e,f)
# res
# inlineNormalize(el,poww, a,b,c,d,e,f) 
#             /(inlineNormalize(a,poww, a,b,c,d,e,f) 
#             +inlineNormalize(b,poww, a,b,c,d,e,f) 
#             +inlineNormalize(c,poww, a,b,c,d,e,f) 
#             +inlineNormalize(d,poww, a,b,c,d,e,f) +
#             inlineNormalize(e,poww, a,b,c,d,e,f) 
#             +inlineNormalize(f,poww, a,b,c,d,e,f) )

# res>1





# a=200
# b=170
# c=10

# pa=1.1
# pb=1.8
# pc=1.2

# function mySmallSoftMax(a,b,c,pa,pb,pc)
#     #normalize
#     sumP=((pa^6)+(pb^6)+(pc^6))
    
#     # sumEl=((a^3)+(b^3)+(c^3))
    
#     sumP=((pa)+(pb)+(pc))
#     sumEl=((a)+(b)+(c))
    
#     return ((a^3)*(pa^6))+((b^3)*(pb^6))+((c^3)*(pc^6))

# end#mySmallSoftMax

# a=0.0001+1
# b=0.2+1
# c=0.999+1


# a=70
# b=4
# c=50

# poww=40
# summ=a+b+c
# aa=(a/summ)+1
# ba=(b/summ)+1
# ca=(c/summ)+1

# ab= aa^poww/(aa^poww+ba^poww+ca^poww)
# bb= ba^poww/(aa^poww+ba^poww+ca^poww)
# cb= ca^poww/(aa^poww+ba^poww+ca^poww)

# ab+bb+cb

# ab*a +b*bb +c*cb

# a=a+1


# poww=4
# as= (a)^poww/((a)^poww+(b)^poww+(c)^poww)
# bs= (b)^poww/((a)^poww+(b)^poww+(c)^poww)
# cs= (c)^poww/((a)^poww+(b)^poww+(c)^poww)

# as+bs+cs
# cs<0


# as= (a)^2/((a)^2+(b)^2+(c)^2)*a
# bs= (b)^2/((a)^2+(b)^2+(c)^2)*b
# cs= (c)^2/((a)^2+(b)^2+(c)^2)*c

# a-b
# b-a

# c=10
# poww=6
# multi=5000


# #given it is always at leas 2
# za=  a^4-(a-b)^4
# zb=  b^4-(a-b)^4


# dda=za/(za+zb)
# ddb=zb/(za+zb)

# za=a-(b-a)
# zb=b-(b-a)



# as=a/(a+b)
# bs=b/(a+b)

# as+bs

# bs=c/(a+b+c)

# dias=


# as+bs+cs


# bs>1


# as=(a*multi)/(a*multi+b*multi+c*multi)
# bs=(b*multi)/(a*multi+b*multi+c*multi)
# cs=(c*multi)/(a*multi+b*multi+c*multi)





# a=70
# b=69
# poww=4
# as= (a)^poww/((a)^poww+(b)^poww+(c)^poww)
# bs= (b)^poww/((a)^poww+(b)^poww+(c)^poww)

# as+bs



# a=70
# b=69
# poww=4
# as= (a/(a+b+c))
# bs= (b/(a+b+c))

# as+bs
