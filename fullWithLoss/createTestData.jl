

# nums=Float32.(rand(1.0:1000000,3,3,3))

"""
get test data for differentiable clustering
"""
function createTestData(Nx, Ny, Nz, oneSidePad, crossBorderWhere)
    totalPad = oneSidePad * 2
    nums = Float32.(reshape(collect(1:Nx*Ny*Nz), (Nx, Ny, Nz)))#./100
    #nums = Float32.(rand(1.0:1000000,Nx,Ny,Nz))

    withPad = Float32.(zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad))
    withPad[(oneSidePad+1):((oneSidePad+Nx)), (oneSidePad+1):(oneSidePad+Ny), (oneSidePad+1):(oneSidePad+Nz)] = nums
    A = CuArray(withPad)
    dA = similar(A)
    probs = Float32.(ones(Nx, Ny, Nz)) .* 0.1

    probs[crossBorderWhere, :, :] .= 0.9
    probs[:, crossBorderWhere, :] .= 0.9
    probs[:, :, crossBorderWhere] .= 0.9
    probsB = ones(Nx, Ny, Nz)
    probs = probsB .- probs#so we will keep low probability on edges
    withPadp = Float32.(zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad))
    withPadp[(oneSidePad+1):((oneSidePad+Nx)), (oneSidePad+1):(oneSidePad+Ny), (oneSidePad+1):(oneSidePad+Nz)] = probs
    dp = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)
    Aout = CUDA.zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad)
    dAout = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)
    dA .= 1
    p = CuArray(withPadp)
    return (A, dA, p, dp, Aout, dAout)

end
