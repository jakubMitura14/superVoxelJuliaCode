using CUDA, Enzyme, Test
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl






function mul_kernel(Nx,Ny,Nz,A)
    x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))
    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))
    z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))

    currMax=100000000.0
    
    if (x <= Nx && y<=Ny && z<=Nz)
        currMax=max(currMax)
        #A[x,y,z]*=A[x,y,z]
    end

    # if x <= length(A)
    #     currMax=max(currMax)
    # end

    return nothing
end

function grad_mul_kernel(Nx,Ny,Nz,A, dA)
    Enzyme.autodiff_deferred(mul_kernel,Const(Nx),Const(Ny), Const(Nz), Duplicated(A, dA))
    return nothing
end


blocks= Int(64*64*64/512)
A = CUDA.ones(64,64,64)
@cuda threads=(8*8*8) blocks = blocks mul_kernel(64,64,64,A)
A = CUDA.ones(64,64,64)
dA = similar(A)
dA .= 1
@cuda threads= (8*8*8) blocks=blocks grad_mul_kernel(64,64,64,A, dA)
print(dA)


# @testset "mul_kernel" begin
#     blocks= Int(64*64*64/512)
#     A = CUDA.ones(64,64,64)
#     @cuda threads=512 blocks = blocks mul_kernel(A)
#     A = CUDA.ones(64,64,64)
#     dA = similar(A)
#     dA .= 1
#     @cuda threads= 512 blocks=blocks grad_mul_kernel(A, dA)
    
#     @test all(dA .== 2)
# end










"""
utility macro to iterate in given range around given voxel
"""
macro iterAround(r,Nx,Ny,Nz,ex   )
    return esc(quote
        #for xAdd in -($r):($r)
            x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))+xAdd
            if(x>0 && x<=$Nx)
                #for yAdd in -$r:$r
                    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))+yAdd
                    if(y>0 && y<=$Ny)
                        #for zAdd in -$r:$r
                            z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))+zAdd
                            if(z>0 && z<=$Nz)
                                #if((abs(xAdd)+abs(yAdd)+abs(zAdd)) <=r)
                                    $ex
                                #end 
                            end
                        end
                    end    
        #         end    
        #     end
        # end    
    end)
end

function mul_kernel(Nx,Ny,Nz,r,A)
    currentMax=0.0
    x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))
    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))
    z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))
    if(x>0 && x<=Nx  && y>0 && y<=Ny   &&  z>0 && z<=Nz   )
        A[x,y,z] = A[x,y,z]*A[x,y,z]
    end    
    #@iterAround(r,Nx,Ny,Nz, A[x,y,z] = A[x,y,z]*A[x,y,z]) 
    #currentMax= A[x,y,z]
    #CUDA.max(currentMax, A[x,y,z] )

    # i = threadIdx().x
    # if i <= length(A)
    #     A[i] *= A[i]
    # end
    return nothing
end

function grad_mul_kernel(Nx,Ny,Nz,r,A, dA)
    Enzyme.autodiff_deferred(mul_kernel, Const(Nx), Const(Ny), Const(Nz), Const(r),  Duplicated(A, dA))
    return nothing
end
Nx,Ny,Nz=64,64,64
A = CUDA.ones(64,64,64).*2
dA = similar(A)
mainArrSize=(Nx,Ny,Nz)
r=1

@cuda threads=(8,8,8) grad_mul_kernel(Nx,Ny,Nz,r  ,A, dA)
dA

Int(maximum(dA))

@testset "mul_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) mul_kernel(A)
    A = CUDA.ones(64,)
    dA = similar(A)
    dA .= 1
    @cuda threads=(8,8,8) grad_mul_kernel(A, dA)
    @test all(dA .== 2)
end





