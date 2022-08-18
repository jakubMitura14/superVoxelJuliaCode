using CUDA, Enzyme, Test
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl
# and https://enzyme.mit.edu/julia/generated/box/


using CUDA
using Enzyme
using Test

numRays=6
lenRay=3
rays= ( ((-3,0,0),(-2,0,0),(-1,0,0))
    , ((1,0,0),(2,0,0),(3,0,0)) 
    , ((0,1,0),(0,2,0),(0,3,0)) 
    , ((0,-1,0),(0,-2,0),(0,-3,0)) 
    , ((0,0,1),(0,0,2),(0,0,3)) 
    , ((0,0,-1),(0,0,-2),(0,0,-3))  
    
    )

    
    
    





function mul_kernel(Nx,Ny,Nz,A,p,Aout,rays,numRays,lenRay)
    #adding one bewcouse of padding
    x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))+1
    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))+1
    z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))+1

    if(p[x+1,y,z]< p[x-1,y,z])
        Aout[x,y,z]=A[x,y,z]
    end
    # Aout[x,y,z]=(p[x+1,y,z]< p[x-1,y,z])
    # *(p[x+1,y,z]< p[x,y+1,z])
    # *(p[x+1,y,z]< p[x,y-1,z])
    # *(p[x+1,y,z]< p[x,y,z+1])
    # *(p[x+1,y,z]< p[x,y,z-1])
    # *(A[x,y,z]<A[x+1,y,z] )   *A[x+1,y,z]+
    # (p[x-1,y,z]< p[x+1,y,z])
    # *(p[x-1,y,z]< p[x,y-1,z])
    # *(p[x-1,y,z]< p[x,y+1,z])
    # *(p[x-1,y,z]< p[x,y-1,z])
    # *(p[x-1,y,z]< p[x,y+1,z])
    # *(A[x,y,z]<A[x-1,y,z] )    *A[x-1,y,z]
    # +
    # (p[x,y+1,z]< p[x+1,y,z])
    # *(p[x,y+1,z]< p[x-1,y,z])
    # *(p[x,y+1,z]< p[x,y-1,z])
    # *(p[x,y+1,z]< p[x,y,z+1])
    # *(p[x,y+1,z]< p[x,y,z-1])
    # *(A[x,y,z]<A[x,y+1,z] )    *A[x,y+1,z]+
    # (p[x,y-1,z]< p[x+1,y,z])
    # *(p[x,y-1,z]< p[x-1,y,z])
    # *(p[x,y-1,z]< p[x,y+1,z])
    # *(p[x,y-1,z]< p[x,y,z+1])
    # *(p[x,y-1,z]< p[x,y,z-1])
    # *(A[x,y,z]<A[x,y-1,z] )   *A[x,y-1,z]+
    # (p[x,y,z+1]< p[x+1,y,z])
    # *(p[x,y,z+1]< p[x-1,y,z])
    # *(p[x,y,z+1]< p[x,y+1,z])
    # *(p[x,y,z+1]< p[x,y-1,z])
    # *(p[x,y,z+1]< p[x,y,z-1])
    # *(A[x,y,z]<A[x,y,z+1] )    *A[x,y,z+1]+
    # (p[x,y,z-1]< p[x+1,y,z])
    # *(p[x,y,z-1]< p[x-1,y,z])
    # *(p[x,y,z-1]< p[x,y+1,z])
    # *(p[x,y,z-1]< p[x,y-1,z])
    # *(p[x,y,z-1]< p[x,y,z+1])
    # *(A[x,y,z]<A[x+1,y,z-1] )    *A[x,y,z-1]



    #CUDA.@cuprint "x $(x) y $(y) z $(z) "
    # Aout[x,y,z]=
    #             p[x,y,z]*(p[x,y,z]>A[x,y,z])
    #             +A[x,y,z]*(p[x,y,z]<A[x,y,z])

    # #we will look in some distance in directions around - then we will save as output the max index from the direction with smallest cost
    # #grid_handle = this_grid()
    # currMax::Float32=0.0
    # corrCost::Float32=3.4028235f38 # max float 32 value
    # #iterate over all rays
    # for i in 1:numRays
    #     rayCost::Float32=0.0
    #     rayMax::Float32=0.0
    #     #iterate over voxels in given direction
    #     for j in 1:lenRay
    #         if( (x+rays[i][j][1])>0 && (x+rays[i][j][1])<=Nx 
    #             && (y+rays[i][j][2])>0 && (y+rays[i][j][2])<=Ny  
    #             && (z+rays[i][j][3])>0 && (z+rays[i][j][3])<=Nz )
    #             #acumulating cost
    #             rayCost=rayCost+p[(x+rays[i][j][1]),(y+rays[i][j][2]),(z+rays[i][j][3])]
    #             #saving ray max
    #             if( A[(x+rays[i][j][1]),(y+rays[i][j][2]),(z+rays[i][j][3])] >rayMax )
    #                 rayMax=A[(x+rays[i][j][1]),(y+rays[i][j][2]),(z+rays[i][j][3])]
    #             end    
    #         end# if in range
    #         #if we got smaller cost then current best we will save it
    #         if(rayCost<corrCost)
    #             currMax=rayMax
    #         end    
    #     end# fo len ray  
    # end #for num rays
    # if(x<=Nx && y<=Ny && z<=Nz )
    #     Aout[x,y,z]=currMax*100.0#p[x,y,z]
    # end    
    

    return nothing
end

function grad_mul_kernel(Nx,Ny,Nz,A, dA,p,dp,Aout,dAout,rays,numRays,lenRay)
    Enzyme.autodiff_deferred(mul_kernel, Const, Const(Nx), Const(Ny), Const(Nz)
    , Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout,dAout),Const(rays),
    Const(numRays),Const(lenRay)
    )
    return nothing
end


Nx,Ny,Nz= 64+2,64+2,64+2

A = CuArray(Float32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
dA = similar(A)


p = CuArray(Float32.(rand(1.0:10000.0,Nx,Ny,Nz)))
dp = CUDA.ones(Nx,Ny,Nz)

Aout = CUDA.zeros(Nx,Ny,Nz)
dAout = CUDA.ones(Nx,Ny,Nz)

dA .= 1
@cuda threads= (8,8,8) blocks=(8,8,8) grad_mul_kernel(Nx,Ny,Nz,A, dA,p,dp,Aout,dAout,rays,numRays,lenRay)

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
