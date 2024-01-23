using ModelingToolkit
using LinearAlgebra
# using OrdinaryDiffEq
# using DiffEqJump
using DiffEqSensitivity
using Zygote
using DifferentialEquations
using CUDA
using DiffEqFlux, Flux, OrdinaryDiffEq, DiffEqSensitivity
using OrdinaryDiffEq

#based on https://discourse.julialang.org/t/diffusion-by-using-cuda-jl-differentialequations-and-diffeqgpu/78310/5





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

function diffuse!(du, u, p,t)
    copy_u= copy(u)
    iter_around_single(copy_u,p)#mutating copy of u
    du[:,:,:]= copy_u.-u #getting diffrence 
end

const Nx = 64                                                                               
const Ny = 64                                                                               
const Nz = 64                                                                             

mask_arr_cpu = ones(Nx,Ny,Nz)
mask_arr_cpu[:,:,50].=0
mask_arr= CuArray(mask_arr_cpu)
how_many_to_offset_curr_max=1
dimToOffset=3


mask_arr_cpu = ones(Nx,Ny,Nz)
mask_arr_cpu[:,:,50].=0
mask_arr_cpu[:,50,:].=0
mask_arr_cpu[50,:,:].=0

u_GPU=Float32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz)))
mask_arr= Float32.(mask_arr_cpu)
# u_GPU=CuArray(Float32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
# mask_arr= CuArray(Float32.(mask_arr_cpu))
# u_GPU=CuArray(Float32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
# mask_arr= CuArray(Float32.(mask_arr_cpu))

# indexArrB=CuArray(Float32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz))))
du_GPU = similar(u_GPU)     


# indexArrA=Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz)))
# mask_arr= Int32.(mask_arr_cpu)
# indexArrB=Int32.(reshape(collect(1:Nx*Ny*Nz),(Nx,Ny,Nz)))

p = mask_arr

diffuse!(du_GPU, u_GPU,0,0)


tspan = (0f0, 200.0)
prob = ODEProblem(diffuse!, u_GPU, tspan,p)
# steadyProb=SteadyStateProblem(prob)
Δt=1.0
# @time sol = solve(prob,Euler(), dt=Δt,save_everystep=false,save_start=false) 
# #@time sol = solve(steadyProb,Euler(), dt=Δt,save_everystep=false,save_start=false) 



function sum_of_solution(u0,p)
    _prob = remake(prob,u0=u0,p=p)
    sum(solve(_prob,Euler(),dt=1.0,reltol=1e-6,abstol=1e-6,saveat=1.0,sensealg=ReverseDiffAdjoint()))#
end

@time du01,dp1 = Zygote.gradient(sum_of_solution,u_GPU,p)



# res = sol(200.0)
# res[1]
# res[2]
# res[100]

# ress=res.*mask_arr
# unique(ress)







