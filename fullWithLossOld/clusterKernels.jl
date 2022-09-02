using CUDA, Enzyme, Test, Plots
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl
# and https://enzyme.mit.edu/julia/generated/box/

# Pk.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")




function expandKernel(Nx, Ny, Nz, A, p, Aout)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #CUDA.@cuprint("x $(x) y $(y) z $(z) curr $(A[x,y,z]) z+1 $(A[x, y, z+1] ) currp $(1 - p[x, y, z]) p in z+1 $(1 - p[x, y, z+1]) ) alamax z +1 $( alaMax(A[x, y, z], ((1 - p[x, y, z+1]) * A[x, y, z+1])) * (1 - p[x, y, z]))  \n    ")
    Aout[x, y, z] = alaMax(A[x, y, z], (A[x+1, y, z]))
    Aout[x, y, z] = alaMax(A[x, y, z], (A[x-1, y, z]))
    Aout[x, y, z] = alaMax(A[x, y, z], (A[x, y+1, z]))
    Aout[x, y, z] = alaMax(A[x, y, z], (A[x, y-1, z]))
    Aout[x, y, z] = alaMax(A[x, y, z], (A[x, y, z+1]))
    Aout[x, y, z] = alaMax(A[x, y, z], (A[x, y, z-1]))


    # curr=A[x, y, z]*p[x,y,z]
    # Aout[x, y, z] = alaMax(curr, (A[x+1, y, z]*p[x+1,y,z])) 
    # Aout[x, y, z] = alaMax(curr, (A[x-1, y, z]*p[x-1,y,z])) 
    # Aout[x, y, z] = alaMax(curr, (A[x, y+1, z]*p[x,y+1,z])) 
    # Aout[x, y, z] = alaMax(curr, (A[x, y-1, z]*p[x,y-1,z]))
    # Aout[x, y, z] = alaMax(curr, (A[x, y, z+1]*p[x,y,z+1]))
    # Aout[x, y, z] = alaMax(curr, (A[x, y, z-1]*p[x,y,z-1]))
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
    p[x, y, z] = (((alaMax(Float32(p[x, y, z]), Float32(0.5)) - 0.48) / 0.52))#*1.2
    #sync_threads()
    #p[x, y, z]+=1#TODO(remove)
    #CUDA.@atomic p[x, y, z] += Float32(1)  #TODO(remove)
    return nothing
end

function scaleDownKernDeffP(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    Enzyme.autodiff_deferred(scaleDownP, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout)
    )
    return nothing
end


function normalizeKern(Nx, Ny, Nz, A, p, Aout, maxx)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #CUDA.@cuprint("x $(x) y $(y) z $(z) curr $(A[x,y,z]) z+1 $(A[x, y, z+1] ) currp $(1 - p[x, y, z]) p in z+1 $(1 - p[x, y, z+1]) ) alamax z +1 $( alaMax(A[x, y, z], ((1 - p[x, y, z+1]) * A[x, y, z+1])) * (1 - p[x, y, z]))  \n    ")
    #in case the probability in this spot is low it will be scaled down accordingly we add 10 for numerical stability
    #p[x, y, z]=(((alaMax(Float32(p[x,y,z]),Float32(0.5))-0.48)/0.52))#*1.2
    A[x, y, z] = (((A[x, y, z]) / maxx) * 10000) + 1 * p[x, y, z]
    return nothing
end

function normalizeKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout, maxx)
    Enzyme.autodiff_deferred(normalizeKern, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout), Const(maxx)
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
    #A[x, y, z]=(A[x, y, z]*p[x,y,z])*1.4
    # Aout[x, y, z]=((A[x, y, z]*(alaMax(Float32(p[x,y,z]),Float32(0.5))-0.48)/0.52))
    #Aout[x, y, z]=((A[x, y, z]*(alaMax(Float32(p[x,y,z]),Float32(0.5))-0.48)/0.52))
    A[x, y, z] = (A[x, y, z] * p[x, y, z])

    return nothing
end
function scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    Enzyme.autodiff_deferred(scaleDownKern, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout))
    return nothing
end






##### composing gradients flow
#idea we can set as parameter both initialization of A and parameters pa
# in order to calculate the derivatives in enzyme we would need 
#https://enzyme.mit.edu/julia/generated/box/
#Here I see you zero out input derivatives each step and initialize output derivatives of next step with input derivatives of the previous one


#futher plan
"""
add parametrization to the expand function 
so in order to detect edge well we can look in all directions of intrest and in each case
    choose line perpendicular to the chosen direction
    next calculate the diffrences accross this line 
    and second order diffrences along it
    additionally on the opposite site we can get mean intensity and variance of the region behind
    idea is that in one direction there can be edge and in other the same region as a current voxel

    we need to add set of learnable parameters to enable the function to learn
        better this learning could be I suppose represented as a polynomial with some growing powers or sth like that, maybe prelu?
        



"""