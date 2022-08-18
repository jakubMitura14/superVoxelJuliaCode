using CUDA, Enzyme, Test
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl
# and https://enzyme.mit.edu/julia/generated/box/


using CUDA
using Enzyme
using Test



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


# @inline function myDiv(a::Float32, b::Float32)::Float32
#     return @myPowTwo((Float32(a) / (Float32(a) + Float32(b))) + 1, 4)
# end

# @inline function normPair(el::Float32, a::Float32, b::Float32)::Float32
#     return @myPowTwo((el / (a + b)) + 1, 4)
# end#normPair

"""
given 2 numbers return sth like max
"""
@inline function alaMax(a::Float32, b::Float32)::Float32
    return (((@myPowTwo((a / (a + b)) + 1, 6) / @myPowTwo((a / (a + b)) + 1, 6) + @myPowTwo((b / (a + b)) + 1, 6))) * a) + ((@myPowTwo((b / (a + b)) + 1, 6) / @myPowTwo((a / (a + b)) + 1, 6) + @myPowTwo((b / (a + b)) + 1, 6)) * b)
end#alaMax

# @inline function alaMaxp(a, b)::Float32
#     # return ((normPair(a,a,b) /(normPair(a,a,b) + normPair(b,a,b) ))) + ((normPair(b,a,b) /(normPair(a,a,b) + normPair(b,a,b) )))
#     return ((@myPowTwo((a / (a + b)) + 1, 4) / @myPowTwo((a / (a + b)) + 1, 4) + @myPowTwo((b / (a + b)) + 1, 4))) + (@myPowTwo((b / (a + b)) + 1, 4) / @myPowTwo((a / (a + b)) + 1, 4) + @myPowTwo((b / (a + b)) + 1, 4))
# end#alaMax


function mul_kernel(Nx, Ny, Nz, A, p, Aout)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    CUDA.@cuprint("x $(x) y $(y) z $(z) curr $(A[x,y,z]) z+1 $(A[x, y, z+1] ) alamax z +1 $( alaMax(A[x, y, z], ((1 - p[x, y, z-1]) * A[x, y, z-1])) * (1 - p[x, y, z]))  \n    ")
    # Aout[x, y, z] = alaMax(A[x, y, z], ((1 - p[x+1, y, z]) * A[x+1, y, z])) * (1 - p[x, y, z])
    # Aout[x, y, z] = alaMax(A[x, y, z], ((1 - p[x-1, y, z]) * A[x-1, y, z])) * (1 - p[x, y, z])
    # Aout[x, y, z] = alaMax(A[x, y, z], ((1 - p[x, y+1, z]) * A[x, y+1, z])) * (1 - p[x, y, z])
    # Aout[x, y, z] = alaMax(A[x, y, z], ((1 - p[x, y-1, z]) * A[x, y-1, z])) * (1 - p[x, y, z])
    Aout[x, y, z] = alaMax(A[x, y, z], ((1 - p[x, y, z+1]) * A[x, y, z+1])) * (1 - p[x, y, z])
    Aout[x, y, z] = alaMax(A[x, y, z], ((1 - p[x, y, z-1]) * A[x, y, z-1])) * (1 - p[x, y, z])

    # alaMaxp(p[x, y, z], p[x+1, y, z]) 
    # (@myPowTwo((p[x+1,y,z]/(p[x+1,y,z]+p[x,y,z]))+1,4))/(@myPowTwo((p[x+1,y,z]/(p[x+1,y,z]+p[x,y,z]))+1,4)+@myPowTwo((p[x,y,z]/(p[x+1,y,z]+p[x,y,z]))+1,4)  )

    #alaMaxp(p[x,y,z],p[x+1,y,z])#*alaMax(p[x,y,z],p[x+1,y,z])

    return nothing
end

function grad_mul_kernel(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    Enzyme.autodiff_deferred(mul_kernel, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout)
    )
    return nothing
end


Nx, Ny, Nz = 8 + 2, 8 + 2, 8 + 2

nums = Float32.(reshape(collect(1:8*8*8), (8, 8, 8)))
withPad= Float32.(zeros(Nx,Ny,Nz))
withPad[2:9,2:9,2:9]=nums

A = CuArray(withPad)
dA = similar(A)
probs = Float32.(ones(8,8,8)).*0.1
probs[4,:,:].=0.9
probs[:,4,:].=0.9
probs[:,:,4].=0.9
withPadp= Float32.(zeros(Nx,Ny,Nz))
withPadp[2:9,2:9,2:9]=probs


p = CuArray(withPadp)
dp = CUDA.ones(Nx, Ny, Nz)

Aout = CUDA.zeros(Nx, Ny, Nz)
dAout = CUDA.ones(Nx, Ny, Nz)

dA .= 1
# @cuda threads = (4, 4, 4) blocks = (2, 2, 2) grad_mul_kernel(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
@cuda threads = (2, 2, 2) blocks = (1, 1, 1) grad_mul_kernel(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)

print("max $(maximum(dp)) min $(minimum(dp))")



numsRef = reshape(collect(1:8*8*8), (8, 8, 8))
withPadRef= zeros(Int,Nx,Ny,Nz)
withPadRef[2:9,2:9,2:9]=numsRef

function compare(x,y,z)
    print("in ref $(withPadRef[x,y,z]) in out int $(Int(round(Aout[x,y,z]))) float $(Aout[x,y,z])  ")
end
compare(3,3,3)
# 3,3,3 54359.566
Int(round(Aout[3,2,2]))
withPadRef[3,2,3]


maximum(Aout)


# /home/jakub/julia-1.7.3-linux-x86_64/julia-1.7.3/bin/julia /media/jakub/NewVolume/projects/superVoxelJuliaCode/ODE_pde_play/enzymePlayE.jl
#/media/jakub/NewVolume/projects/superVoxelJuliaCode/ODE_pde_play/enzymePlayE.jl