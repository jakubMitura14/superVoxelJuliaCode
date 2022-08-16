using ModelingToolkit
using LinearAlgebra
# using OrdinaryDiffEq
# using DiffEqJump
using DiffEqSensitivity
using Zygote
using DifferentialEquations
using CUDA
using DiffEqFlux, Flux, OrdinaryDiffEq, DiffEqSensitivity

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


mask_arr_cpu = ones(Nx,Ny,Nz)
mask_arr_cpu[:,:,50].=0
mask_arr_cpu[:,50,:].=0
mask_arr_cpu[50,:,:].=0
# indexArrA=CuArray(Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
# mask_arr= CuArray(Int32.(mask_arr_cpu))
# indexArrB=CuArray(Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))


# indexArrA=Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz)))
# mask_arr= Int32.(mask_arr_cpu)
# indexArrB=Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz)))

p = mask_arr
rate1(u,p,t) = 0.5  # β*S*I
tspan=(0.0, 150.0)
function affect1!(integrator)
  iter_around_single(integrator.u,integrator.p)
end
jump = ConstantRateJump(rate1,affect1!)
u0=indexArrA
prob = DiscreteProblem(u0, tspan, p)
jump_prob = JumpProblem(prob, Direct(), jump)

sol = solve(jump_prob, SSAStepper())
res=sol(150.0)
ress=res.*mask_arr
unique(ress)


function sum_of_solution(u₀,p)
    _prob = remake(prob,u0=u₀,p=p)
    sum(solve(_prob, SSAStepper()))
end

####TODO()
#try with less dense soluton saves with adjoints
# https://diffeq.sciml.ai/stable/analysis/sensitivity/#Example-controlling-adjoint-method-choices-and-checkpointing
#


####
@time du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)
