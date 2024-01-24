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
function save_max(arr_stay,arr_move,mask_arr, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    
    view_stay=offset_end(arr_stay, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    view_move=offset_beg(arr_move.* mask_arr, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    
    @views @. view_stay[:,:,:]=max.(view_stay,view_move)

    view_stay=offset_beg(arr_stay, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    view_move=offset_end(arr_move.* mask_arr, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    @views @. view_stay[:,:,:]=max.(view_stay,view_move)
end#offset_end


"""
get_accumulated_maskArr helper function
"""
function save_mask_prim(arr_stay,arr_move, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    view_stay=offset_end(arr_stay, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    view_move=offset_beg(arr_move, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    
    @views @. view_stay[:,:,:]=(view_stay.*view_move)

    view_stay=offset_beg(arr_stay, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    view_move=offset_end(arr_move, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    @views @. view_stay[:,:,:]=(view_stay.*view_move)
end#offset_end


"""
creates adaptation of mask array for bigger translations - that will prohibit jumping over the border
    in index array
"""
function get_accumulated_maskArr(mask_arr, how_many_to_offset_curr_max,dimToOffset,dimX,dimY,dimZ)
    mask_arr_in = copy(mask_arr)
    for how_many_to_offset in 1:(how_many_to_offset_curr_max-1)
        save_mask_prim(mask_arr_in,mask_arr_in, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)

        #print(" how_many_to_offset $(how_many_to_offset) dimToOffset $(dimToOffset)")
        #save_max(arr_stay,arr_move,mask_arr, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    end
    return mask_arr_in
end#get_accumulated_maskArr    

# moddedMask=get_accumulated_maskArr(mask_arr,how_many_to_offset_curr_max,dimToOffset,Nx,Ny,Nz)

# moddedMask[1,1,50]
# mask_arr_cpu[1,1,51]




# sum(moddedMask)
# sum(mask_arr_cpu)

"""
iterate in all 9 directions and with all set offsets
"""
function iter_around(arr_stay,arr_move,mask_arr, how_many_to_offset_max,dimX,dimY,dimZ)
    for how_many_to_offset in 1:how_many_to_offset_max, dimToOffset in [1,2,3]
        moddedMask=get_accumulated_maskArr(mask_arr,how_many_to_offset_curr_max,dimToOffset,Nx,Ny,Nz)
        #print(" how_many_to_offset $(how_many_to_offset) dimToOffset $(dimToOffset)")
        save_max(arr_stay,arr_move,moddedMask, dimToOffset, how_many_to_offset,dimX,dimY,dimZ)
    end
end#iter_around    



"""
iterate in all 9 directions and with all set offsets just one in each direction at a time
"""
function iter_around_single(arr_stay,mask_arr)
    dimX,dimY,dimZ= size(arr_stay)
    for dimToOffset in [1,2,3]
        # moddedMask=get_accumulated_maskArr(mask_arr,how_many_to_offset_curr_max,dimToOffset,Nx,Ny,Nz)
        #print(" how_many_to_offset $(how_many_to_offset) dimToOffset $(dimToOffset)")
        save_max(arr_stay,arr_stay,mask_arr, dimToOffset, 1,dimX,dimY,dimZ)
    end
end#iter_around   

const Nx = 128                                                                               
const Ny = 128                                                                               
const Nz = 128                                                                             

mask_arr_cpu = ones(Nx,Ny,Nz)
mask_arr_cpu[:,:,50].=0
mask_arr= CuArray(mask_arr_cpu)
how_many_to_offset_curr_max=1
dimToOffset=3


indexArrA=CuArray(Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
mask_arr_cpu = ones(Nx,Ny,Nz)
mask_arr_cpu[:,:,50].=0
mask_arr_cpu[:,50,:].=0
mask_arr_cpu[50,:,:].=0
mask_arr= CuArray(Int32.(mask_arr_cpu))
indexArrB=CuArray(Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))



# how_many_to_offset_max=2
# for i in 1:300
#     iter_around_single(indexArrA,mask_arr, Nx,Ny,Nz)
# end


using ModelingToolkit
using LinearAlgebra
using OrdinaryDiffEq


indexArrA[127,127,127]
indexArrA[3,1,1]
indexArrA[126,59,1]

unique(indexArrA.*mask_arr)


indexArrA[4]

p=mask_arr

u0=indexArrA
tspan = (0.0, 300.0)
f(u,p,t) = iter_around_single(u,p)
sys = DiscreteProblem(f, u0, tspan,p)
sol = solve(sys, FunctionMap());




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

