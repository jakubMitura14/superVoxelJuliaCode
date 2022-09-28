using CUDA, Enzyme, Test
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl
# and https://enzyme.mit.edu/julia/generated/box/


using CUDA
using Enzyme
using Test
using Plots


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

@inline function normPaira(a::Float32, b::Float32)::Float32
    return @myPowTwo((a / (a + b)) + 1, 5)
end#normPair

@inline function normPairb(a::Float32, b::Float32)::Float32
    return @myPowTwo((b / (a + b)) + 1, 5)
end#normPair

"""
given 2 numbers return sth like max
"""
@inline function alaMax(a, b)::Float32
    return ((normPaira(a, b) / (normPaira(a, b) + normPairb(a, b))) * a) + ((normPairb(a, b) / (normPaira(a, b) + normPairb(a, b))) * b)
end#alaMax

@inline function alaMaxp(a, b)::Float32
    return ((normPaira(a, b) / (normPaira(a, b) + normPairb(a, b)))) + ((normPairb(a, b) / (normPaira(a, b) + normPairb(a, b))))
end#alaMax


a = 0.55
((alaMax(Float32(a), Float32(0.5)) - 0.48) / 0.52)


# alaMax(Float32(100.0),Float32(10.0))


# @inline function alaMaxp(a, b)::Float32
#     # return ((normPair(a,a,b) /(normPair(a,a,b) + normPair(b,a,b) ))) + ((normPair(b,a,b) /(normPair(a,a,b) + normPair(b,a,b) )))
#     return ((@myPowTwo((a / (a + b)) + 1, 4) / @myPowTwo((a / (a + b)) + 1, 4) + @myPowTwo((b / (a + b)) + 1, 4))) + (@myPowTwo((b / (a + b)) + 1, 4) / @myPowTwo((a / (a + b)) + 1, 4) + @myPowTwo((b / (a + b)) + 1, 4))
# end#alaMax


@inline function getScaled(a::Float32, pp::Float32)::Float32
    return ((a * (alaMax(Float32(pp), Float32(0.5)) - 0.48) / 0.52)) * 10
    #return alaMax(pp,Float32(0.5))
end
# Aout[x, y, z]=((a*(alaMax(Float32(pp),Float32(0.5))-0.48)/0.52))*4


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

    curr::Float32 = getScaled(A[x, y, z], p[x, y, z]) #(((A[x, y, z]*(alaMax(Float32(p[x,y,z]),Float32(0.5))-0.48)/0.52))*4)
    Aout[x, y, z] = alaMax(curr, getScaled(A[x+1, y, z], p[x+1, y, z]))
    Aout[x, y, z] = alaMax(curr, getScaled(A[x-1, y, z], p[x-1, y, z]))
    Aout[x, y, z] = alaMax(curr, getScaled(A[x, y+1, z], p[x, y+1, z]))
    Aout[x, y, z] = alaMax(curr, getScaled(A[x, y-1, z], p[x, y-1, z]))
    Aout[x, y, z] = alaMax(curr, getScaled(A[x, y, z+1], p[x, y, z+1]))
    Aout[x, y, z] = alaMax(curr, getScaled(A[x, y, z-1], p[x, y, z-1]))
    Aout[x, y, z] = getScaled(Aout[x, y, z], p[x, y, z-1])



    # (((A[x, y, z]*(alaMax(Float32(p[x,y,z]),Float32(0.5))-0.48)/0.52))*4)

    #Aout[x, y, z] = (A[x, y, z]+(A[x+1, y, z])+(A[x-1, y, z])+(A[x, y+1, z])+(A[x, y-1, z])+(A[x, y, z+1])+(A[x, y, z-1]) )/7


    return nothing
end

function expandKernelDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    Enzyme.autodiff_deferred(expandKernel, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout)
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
    Aout[x, y, z] = ((A[x, y, z] * (alaMax(Float32(p[x, y, z]), Float32(0.5)) - 0.48) / 0.52)) * 2

    return nothing
end
function scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    Enzyme.autodiff_deferred(scaleDownKern, Const, Const(Nx), Const(Ny), Const(Nz), Duplicated(A, dA), Duplicated(p, dp), Duplicated(Aout, dAout)
    )
    return nothing
end



Nx, Ny, Nz = 8 + 2, 8 + 2, 8 + 2

nums = Float32.(reshape(collect(1:8*8*8), (8, 8, 8)))

#nums=Float32.(rand(1:10000, 8,8,8))

withPad = Float32.(zeros(Nx, Ny, Nz))
withPad[2:9, 2:9, 2:9] = nums

A = CuArray(withPad)
dA = similar(A)
probs = Float32.(ones(8, 8, 8)) .* 0.1
probs[4, :, :] .= 0.9
probs[:, 4, :] .= 0.9
probs[:, :, 4] .= 0.9
probsB = ones(8, 8, 8)
probs = probsB .- probs#so we will keep low probability on edges
withPadp = Float32.(zeros(Nx, Ny, Nz))
withPadp[2:9, 2:9, 2:9] = probs

Int(round(maximum(withPadp)))

p = CuArray(withPadp)
dp = CUDA.ones(Nx, Ny, Nz)

Aout = CUDA.zeros(Nx, Ny, Nz)
dAout = CUDA.ones(Nx, Ny, Nz)

dA .= 1
@cuda threads = (4, 4, 4) blocks = (2, 2, 2) expandKernelDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
@cuda threads = (4, 4, 4) blocks = (2, 2, 2) scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)

#@cuda threads = (2, 2, 2) blocks = (1, 1, 1) grad_mul_kernel(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)

print("max $(maximum(dp)) min $(minimum(dp))")



numsRef = reshape(collect(1:8*8*8), (8, 8, 8))
withPadRef = zeros(Int, Nx, Ny, Nz)
withPadRef[2:9, 2:9, 2:9] = numsRef

function compare(x, y, z)
    print("in ref $(withPadRef[x,y,z]) in out int $(Int(round(Aout[x,y,z]))) float $(Aout[x,y,z])  ")
end
compare(3, 3, 3)
# 3,3,3 54359.566
Int(round(Aout[3, 3, 4]))
withPadRef[3, 3, 5]
p[3, 3, 6]

cpuArr = Array(Aout[3, :, :])
cpuArrPrim = Array(A[3, :, :])
# heatmap(cpuArrPrim)
# heatmap(cpuArr)

for i in 1:5
    #@cuda threads = (4, 4, 4) blocks = (2, 2, 2) scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    #A=Aout
    @cuda threads = (4, 4, 4) blocks = (2, 2, 2) expandKernelDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    #@cuda threads = (4, 4, 4) blocks = (2, 2, 2) scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
    A = Aout
end

cpuArr = Array(Aout[3, :, :])
topLeft = cpuArr[2, 2]
topRight = cpuArr[9, 2]
bottomLeft = cpuArr[2, 9]
bottomRight = cpuArr[9, 9]

topLeftCorn = Array(Aout[3, :, :])
topRightCorn = Array(Aout[3, :, :])
bottomLeftCorn = Array(Aout[3, :, :])
bottomRightCorn = Array(Aout[3, :, :])


heatmap(cpuArr)
print("topLeft $(topLeft) topRight $(topRight) bottomLeft $(bottomLeft)  bottomRight $(bottomRight)")


Array(Aout[3, :, :])
Array(Aout[4, :, :])
Array(Aout[5, :, :])
Array(Aout[6, :, :])
Array(Aout[7, :, :])

maximum(Aout)

###############
#scale


((a - 0.5) * 10000) + ((a - 0.5) * 100)^2
((b - 0.5) * 10000) + ((b - 0.5) * 100)^2


a = 0.45
b = 0.55
a = 0.01
b = 0.99

a = 0.6
b = 0.7



as = ((a - 0.5) + (b - 0.5))
bs = ((b - 0.5) + (b - 0.5))
aa = as / (as + bs)
bb = bs / (as + bs)


(((as) + 500) + 1) / 1000
(((bs) + 500) + 1) / 1000

a = 0.0000001#-48
a = 0.01#-50
a = 0.1#-38
a = 0.45#-3
a = 0.55#7
a = 0.7#22
a = 0.99#52
a = 0.999999#53
(((((a - 0.5) * 100 + (a + 0.5)^2) + 1)) / 54)

a = 0.3
(a - 0.5) / (a - 0.5)


zz = (a^2 - 2 * a * b - b^2) / (a^2 - 2 * a * b + b^2)


a = 0.55
b = 0.5
((a - b) + 1)^3

ass = a - 0.5
asb = ((ass + 1)^2)
asc = (ass * 10) + 5


# /home/jakub/julia-1.7.3-linux-x86_64/julia-1.7.3/bin/julia /media/jakub/NewVolume/projects/superVoxelJuliaCode/ODE_pde_play/enzymePlayE.jl
#/media/jakub/NewVolume/projects/superVoxelJuliaCode/ODE_pde_play/enzymePlayE.jl