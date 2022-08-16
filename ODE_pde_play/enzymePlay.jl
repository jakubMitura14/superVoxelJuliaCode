using CUDA, Enzyme, Test
# based on https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl



function max_kernel(A,B)
    i = (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))
    if i <= length(A)
        A[i] =max(A[i],B[i])
    end
    return nothing
end

function elWiseCPUMax(A,B )
    for i in eachindex(A)
        A[i] =max(A[i],B[i])
    
    end #for    
end
A=zeros(8,8,8)
B=ones(8,8,8)
elWiseCPUMax(A,B)
A


"""
takes a view of given array translates it given number of times
arrToOfset - array from which we create view
dimToOffset - in which dimension we want to translate array
how_many_to_offset - how big the translation should be
dimX,dimY,dimZ - dimensions of the arrToOfset

offset_beg - offsets from the begining offset_end offsets from the end
"""
function offset_beg(arrToOfset, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    if(dimToOffset==1)
        return  view(arrToOfset, (how_many_to_offset+1) :dimX, 1:dimY, 1:dimZ)
    end
    if(dimToOffset==2)
        return  view(arrToOfset, 1:dimX, (how_many_to_offset+1):dimY, 1:dimZ)

    end
    if(dimToOffset==3)
        return  view(arrToOfset, 1:dimX, 1:dimY, (how_many_to_offset+1):dimZ)

    end    
end#offset_beg

function offset_end(arrToOfset, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    if(dimToOffset==1)
        return  view(arrToOfset, 1:dimX-(how_many_to_offset), 1:dimY, 1:dimZ)
    end
    if(dimToOffset==2)
        return  view(arrToOfset, 1:dimX,1:dimY-(how_many_to_offset), 1:dimZ)

    end
    if(dimToOffset==3)
        return  view(arrToOfset, 1:dimX, 1:dimY, 1:dimZ-(how_many_to_offset))

    end    
end#offset_end




"""
comapre array_stay to translated arr_move and mutates arr_stay with maximum of it and translated ar_move 
arr_stay - array that stays in place - we will offset its end and we will mutate it
arr_move - array which we move we offset its end
"""
function save_max(arr_stay,arr_move, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
  
    view_stay=offset_end(arr_stay, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    view_move=offset_beg(arr_move, dimToOffset, how_many_to_offset,dimX,dimY,dimZ) 
    @views @. view_stay[:,:,:]=elWiseCPUMax(view_stay,view_move)

end#offset_end




"""
iterate in all 9 directions and with all set offsets
"""
function iter_around(arr_stay,arr_move, how_many_to_offset_max,dimX,dimY,dimZ)
    for how_many_to_offset in 1:how_many_to_offset_max, dimToOffset in [1,2,3]
        #print(" how_many_to_offset $(how_many_to_offset) dimToOffset $(dimToOffset)")
        save_max(arr_stay,arr_move, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    end
end#iter_around    

"""
iterate in all 9 directions and with all set offsets just one in each direction at a time
"""
function iter_around_single(arr_stay)
    dimX,dimY,dimZ= size(arr_stay)
    for dimToOffset in [1,2,3]
        # moddedMask=get_accumulated_maskArr(mask_arr,how_many_to_offset_curr_max,dimToOffset,Nx,Ny,Nz)
        #print(" how_many_to_offset $(how_many_to_offset) dimToOffset $(dimToOffset)")
        save_max(arr_stay,arr_stay, dimToOffset, 1,dimX,dimY,dimZ)
    end
end#iter_around   


const Nx = 128                                                                               
const Ny = 128                                                                               
const Nz = 128                                                                             

mask_arr_cpu = ones(Nx,Ny,Nz)
mask_arr_cpu[:,:,50].=0
mask_arr= mask_arr_cpu
how_many_to_offset_curr_max=1
dimToOffset=3


mask_arr_cpu = ones(Nx,Ny,Nz)
mask_arr_cpu[:,:,50].=0
mask_arr_cpu[:,50,:].=0
mask_arr_cpu[50,:,:].=0
indexArrA=Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz)))

d_indexArrA=similar(indexArrA)
Enzyme.autodiff_deferred(iter_around_single,  Const, Duplicated(indexArrA, d_indexArrA))











"""
utility macro to iterate in given range around given voxel
"""
macro iterAround(r,Nx,Ny,Nz,ex   )
    return esc(quote
        for xAdd in -$r:$r
            x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))+xAdd
            if(x>0 && x<=$Nx)
                for yAdd in -$r:$r
                    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))+yAdd
                    if(y>0 && y<=$Ny)
                        for zAdd in -$r:$r
                            z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))+zAdd
                            if(z>0 && z<=$Nz)
                                if((abs(xAdd)+abs(yAdd)+abs(zAdd)) <=r)
                                    $ex
                                end 
                            end
                        end
                    end    
                end    
            end
        end    
    end)
end

function mul_kernel(Nx,Ny,Nz,r,A)
    currentMax=0.0
    @iterAround(r,Nx,Ny,Nz, currentMax=1.0) 
    #currentMax= A[x,y,z]
    #CUDA.max(currentMax, A[x,y,z] )

    # i = threadIdx().x
    # if i <= length(A)
    #     A[i] *= A[i]
    # end
    return nothing
end

function grad_mul_kernel(Nx,Ny,Nz,r,A, dA)
    Enzyme.autodiff_deferred(mul_kernel, Const, Const, Const, Const, Const, Duplicated(A, dA))
    return nothing
end
Nx,Ny,Nz=64,64,64
A = CUDA.rand(64,64,64)
dA = similar(A)
mainArrSize=(Nx,Ny,Nz)
r=1

@cuda threads=(8,8,8) grad_mul_kernel(Nx,Ny,Nz,r  ,A, dA)
dA


@testset "mul_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) mul_kernel(A)
    A = CUDA.ones(64,)
    dA = similar(A)
    dA .= 1
    @cuda threads=(8,8,8) grad_mul_kernel(A, dA)
    @test all(dA .== 2)
end





