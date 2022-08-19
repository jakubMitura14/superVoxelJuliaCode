using CUDA, Enzyme, Test,Plots
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl
# and https://enzyme.mit.edu/julia/generated/box/


"""
for some reason raising to the power greater then 2 give erro - hence this macro enable caling power of two multiple times
"""
macro myPowTwo(ex, num)
    exIn = quote
        (($ex)^2)
    end
    for i in 1:(num-1)
        exIn = quote
            (($exIn)^2)
        end
    end
    return esc(:($exIn))
end



@inline function normPaira(a::Float32,b::Float32)::Float32
    return @myPowTwo((a/(a+b))+1,5)
end#normPair

@inline function normPairb(a::Float32,b::Float32)::Float32
    return @myPowTwo((b/(a+b))+1,5)
end#normPair

"""
given 2 numbers return sth like max
"""
@inline  function alaMax(a,b)::Float32
   return ((normPaira(a,b) /(normPaira(a,b) + normPairb(a,b) ))*a) + ((normPairb(a,b) /(normPaira(a,b) + normPairb(a,b) ))*b)
end#alaMax


#idea divide in the end by this spot probability and as a preprocessing step multiply A by inversed p squared ...
function expandKernel(Nx, Ny, Nz, A, p, Aout)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #CUDA.@cuprint("x $(x) y $(y) z $(z) curr $(A[x,y,z]) z+1 $(A[x, y, z+1] ) currp $(1 - p[x, y, z]) p in z+1 $(1 - p[x, y, z+1]) ) alamax z +1 $( alaMax(A[x, y, z], ((1 - p[x, y, z+1]) * A[x, y, z+1])) * (1 - p[x, y, z]))  \n    ")
    # Aout[x, y, z] = alaMax(A[x, y, z], (A[x+1, y, z])) 
    # Aout[x, y, z] = alaMax(A[x, y, z], (A[x-1, y, z])) 
    # Aout[x, y, z] = alaMax(A[x, y, z], (A[x, y+1, z])) 
    # Aout[x, y, z] = alaMax(A[x, y, z], (A[x, y-1, z]))
    # Aout[x, y, z] = alaMax(A[x, y, z], (A[x, y, z+1]))
    # Aout[x, y, z] = alaMax(A[x, y, z], (A[x, y, z-1]))

    
    curr=A[x, y, z]*p[x,y,z]
    Aout[x, y, z] = alaMax(curr, (A[x+1, y, z]*p[x+1,y,z])) 
    Aout[x, y, z] = alaMax(curr, (A[x-1, y, z]*p[x-1,y,z])) 
    Aout[x, y, z] = alaMax(curr, (A[x, y+1, z]*p[x,y+1,z])) 
    Aout[x, y, z] = alaMax(curr, (A[x, y-1, z]*p[x,y-1,z]))
    Aout[x, y, z] = alaMax(curr, (A[x, y, z+1]*p[x,y,z+1]))
    Aout[x, y, z] = alaMax(curr, (A[x, y, z-1]*p[x,y,z-1]))
    return nothing
end

function expandKernelDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    Enzyme.autodiff_deferred(expandKernel, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout)
    )
    return nothing
end

function scaleDownP(Nx, Ny, Nz, A, p, Aout)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #CUDA.@cuprint("x $(x) y $(y) z $(z) curr $(A[x,y,z]) z+1 $(A[x, y, z+1] ) currp $(1 - p[x, y, z]) p in z+1 $(1 - p[x, y, z+1]) ) alamax z +1 $( alaMax(A[x, y, z], ((1 - p[x, y, z+1]) * A[x, y, z+1])) * (1 - p[x, y, z]))  \n    ")
    #in case the probability in this spot is low it will be scaled down accordingly we add 10 for numerical stability
    p[x, y, z]=(((alaMax(Float32(p[x,y,z]),Float32(0.5))-0.48)/0.52))#*1.2
   
    return nothing
end

function scaleDownKernDeffP(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    Enzyme.autodiff_deferred(scaleDownP, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout)
    )
    return nothing
end



function scaleDownKern(Nx, Ny, Nz, A, p, Aout)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #CUDA.@cuprint("x $(x) y $(y) z $(z) curr $(A[x,y,z]) z+1 $(A[x, y, z+1] ) currp $(1 - p[x, y, z]) p in z+1 $(1 - p[x, y, z+1]) ) alamax z +1 $( alaMax(A[x, y, z], ((1 - p[x, y, z+1]) * A[x, y, z+1])) * (1 - p[x, y, z]))  \n    ")
    #in case the probability in this spot is low it will be scaled down accordingly we add 10 for numerical stability
    A[x, y, z]=(A[x, y, z]*p[x,y,z])*1.4
   
    return nothing
end
function scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    Enzyme.autodiff_deferred(scaleDownKern, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout) )
    return nothing
end



"""
get test data for differentiable clustering
"""
function createTestData(Nx, Ny, Nz,oneSidePad, crossBorderWhere)
    totalPad=oneSidePad*2
    nums = Float32.(reshape(collect(1:Nx*Ny*Nz), (Nx, Ny, Nz)))#./100
    withPad= Float32.(zeros(Nx+totalPad,Ny+totalPad,Nz+totalPad))
    withPad[(oneSidePad+1):((oneSidePad+Nx)),(oneSidePad+1):(oneSidePad+Ny),(oneSidePad+1):(oneSidePad+Nz)]=nums
    A = CuArray(withPad)
    dA = similar(A)
    probs = Float32.(ones(Nx,Ny,Nz)).*0.1  

    probs[crossBorderWhere,:,:].=0.9
    probs[:,crossBorderWhere,:].=0.9
    probs[:,:,crossBorderWhere].=0.9
    probsB=ones(Nx,Ny,Nz)
    probs=probsB.-probs#so we will keep low probability on edges
    withPadp= Float32.(zeros(Nx+totalPad,Ny+totalPad,Nz+totalPad))
    withPadp[(oneSidePad+1):((oneSidePad+Nx)),(oneSidePad+1):(oneSidePad+Ny),(oneSidePad+1):(oneSidePad+Nz)]=probs
    dp = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad)
    Aout = CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad)
    dAout = CUDA.ones(Nx+totalPad, Ny+totalPad, Nz+totalPad)
    dA .= 1
    p = CuArray(withPadp)
    return (A, dA, p, dp, Aout, dAout)

end    

### test Data

Nx, Ny, Nz = 8 , 8 , 8
oneSidePad=1
crossBorderWhere=8
A, dA, p, dp, Aout, dAout=createTestData(Nx, Ny, Nz,oneSidePad, crossBorderWhere)


### run

@cuda threads = (4, 4, 4) blocks = (2, 2, 2) scaleDownKernDeffP(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
for i in 1:20
    @cuda threads = (4, 4, 4) blocks = (2 ,2, 2) scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    # A=Aout
    @cuda threads = (4, 4, 4) blocks = (2, 2, 2) expandKernelDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    @cuda threads = (4, 4, 4) blocks = (2, 2, 2) scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)

    A=Aout
end
@cuda threads = (4, 4, 4) blocks = (2 ,2, 2) scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)



cpuArr=Array(Aout[3,:,:])
topLeft= cpuArr[2,2]
topRight= cpuArr[9,2]
bottomLeft= cpuArr[2,9]
bottomRight= cpuArr[9,9]

topLeftCorn= Array(Aout[3,:,:])
topRightCorn= Array(Aout[3,:,:])
bottomLeftCorn=Array(Aout[3,:,:])
bottomRightCorn= Array(Aout[3,:,:])


heatmap(cpuArr)
print("topLeft $(topLeft) topRight $(topRight) bottomLeft $(bottomLeft)  bottomRight $(bottomRight)")


