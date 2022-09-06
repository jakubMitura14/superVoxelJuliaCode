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


@inline function myDiv(a::Float32,b::Float32)::Float32
    return @myPowTwo((Float32(a)/(Float32(a)+Float32(b)))+1,4)
end 

@inline function normPair(el::Float32,a::Float32,b::Float32)::Float32
    return @myPowTwo((el/(a+b))+1,4)
end#normPair

"""
given 2 numbers return sth like max
"""
@inline  function alaMax(a,b)::Float32
    return (((@myPowTwo((a/(a+b))+1,4) /@myPowTwo((a/(a+b))+1,4) + @myPowTwo((b/(a+b))+1,4))) *a)+ ((@myPowTwo((b/(a+b))+1,4) /@myPowTwo((a/(a+b))+1,4) + @myPowTwo((b/(a+b))+1,4))*b)
end#alaMax

@inline  function alaMaxp(a,b)::Float32
    # return ((normPair(a,a,b) /(normPair(a,a,b) + normPair(b,a,b) ))) + ((normPair(b,a,b) /(normPair(a,a,b) + normPair(b,a,b) )))
    return ((@myPowTwo((a/(a+b))+1,8) /@myPowTwo((a/(a+b))+1,8) + @myPowTwo((b/(a+b))+1,8))) + (@myPowTwo((b/(a+b))+1,8) /@myPowTwo((a/(a+b))+1,8) + @myPowTwo((b/(a+b))+1,8))
 end#alaMax


function mul_kernel(Nx,Ny,Nz,A,p,Aout)
    #adding one bewcouse of padding
    x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))+1
    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))+1
    z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))+1

    Aout[x,y,z]=alaMaxp(p[x,y,z],p[x+1,y,z]) *alaMax(A[x,y,z],A[x+1,y,z])
    # (@myPowTwo((p[x+1,y,z]/(p[x+1,y,z]+p[x,y,z]))+1,4))/(@myPowTwo((p[x+1,y,z]/(p[x+1,y,z]+p[x,y,z]))+1,4)+@myPowTwo((p[x,y,z]/(p[x+1,y,z]+p[x,y,z]))+1,4)  )

    #alaMaxp(p[x,y,z],p[x+1,y,z])#*alaMax(p[x,y,z],p[x+1,y,z])
 
    return nothing
end

function grad_mul_kernel(Nx,Ny,Nz,A, dA,p,dp,Aout,dAout)
    Enzyme.autodiff_deferred(mul_kernel, Const, Const(Nx), Const(Ny), Const(Nz)
    , Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout,dAout)
    )
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


a=200
b=170
c=10

pa=1.1
pb=1.8
pc=1.2

function mySmallSoftMax(a,b,c,pa,pb,pc)
    #normalize
    sumP=((pa^6)+(pb^6)+(pc^6))
    
    # sumEl=((a^3)+(b^3)+(c^3))
    
    sumP=((pa)+(pb)+(pc))
    sumEl=((a)+(b)+(c))
    
    return ((a^3)*(pa^6))+((b^3)*(pb^6))+((c^3)*(pc^6))

end#mySmallSoftMax

a=70
b=4

a-b
b-a

c=10
poww=6
multi=5000


#given it is always at leas 2
za=  a^4-(a-b)^4
zb=  b^4-(a-b)^4


dda=za/(za+zb)
ddb=zb/(za+zb)

za=a-(b-a)
zb=b-(b-a)



as=a/(a+b)
bs=b/(a+b)

as+bs

bs=c/(a+b+c)

dias=


as+bs+cs


bs>1


as=(a*multi)/(a*multi+b*multi+c*multi)
bs=(b*multi)/(a*multi+b*multi+c*multi)
cs=(c*multi)/(a*multi+b*multi+c*multi)


as= (a)^poww/((a)^poww+(b)^poww+(c)^poww)
bs= (b)^poww/((a)^poww+(b)^poww+(c)^poww)
cs= (c)^poww/((a*20)^poww+(b*20)^poww+(c*20)^poww)


a=70
b=69
poww=4
as= (a)^poww/((a)^poww+(b)^poww+(c)^poww)
bs= (b)^poww/((a)^poww+(b)^poww+(c)^poww)

as+bs



a=70
b=69
poww=4
as= (a/(a+b+c))
bs= (b/(a+b+c))

as+bs
