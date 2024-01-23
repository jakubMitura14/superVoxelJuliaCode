using CUDA
#from https://discourse.julialang.org/t/diffusion-by-using-cuda-jl-differentialequations-and-diffeqgpu/78310
                                                                      # Width


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
    @views @. view_stay[:,:,:]=max.(view_stay,view_move)

    view_stay=offset_beg(arr_stay, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    view_move=offset_end(arr_move, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    @views @. view_stay[:,:,:]=max.(view_stay,view_move)
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

const Nx = 128                                                                               
const Ny = 128                                                                               
const Nz = 128                                                                             

indexArrA=CuArray(Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
indexArrB=CuArray(Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
indexArrC=CuArray(Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))

how_many_to_offset_max=2
for i in 1:200
    iter_around(indexArrA,indexArrA, how_many_to_offset_max,Nx,Ny,Nz)
end

indexArrA[3]
maximum(indexArrA)
maximum(indexArrB)
maximum(indexArrC)
indexArrB[1]



dimToOffset=3
how_many_to_offset=2
save_max(indexArrA,indexArrB, dimToOffset, how_many_to_offset,Nx,Ny,Nz)
indexArrA[1,1,1]


view_stay=offset_beg(indexArrA, dimToOffset, how_many_to_offset,Nx,Ny,Nz)
view_move=offset_end(indexArrB, dimToOffset, how_many_to_offset,Nx,Ny,Nz)

@views @. view_stay[:,:,:]=max.(view_stay,view_move)

indexArrA[1]
indexArrB[1]


arrMax=max.(view_stay,view_move)
arrMax[1]
view_stay[1]
view_move[1]

