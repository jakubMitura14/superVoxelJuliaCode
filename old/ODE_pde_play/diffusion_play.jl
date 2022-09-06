using ModelingToolkit
using LinearAlgebra
# using OrdinaryDiffEq
# using DiffEqJump
using DiffEqSensitivity
using Zygote
using DifferentialEquations
using CUDA
using DiffEqFlux, Flux, OrdinaryDiffEq, DiffEqSensitivity

#based on https://discourse.julialang.org/t/diffusion-by-using-cuda-jl-differentialequations-and-diffeqgpu/78310/5
const α  = 1f-4                                                                             # Diffusivity
const L  = 1f-1                                                                             # Length
const W  = 1f-1                                                                             # Width
const Nx = 64                                                                               
const Ny = 64                                                                               
const Nz = 64                                                                            # No.of steps in y-axis
const Δx = Float32(L/(Nx-1))                                                                # x-grid spacing
const Δy = Float32(W/(Ny-1))                                                                # y-grid spacing
const Δz = Float32(W/(Nz-1))                                                                # y-grid spacing
const Δt = 1.0  #Float32(Δx^2 * Δy^2 / (2f0 * α * (Δx^2 + Δy^2)))                                 # Largest stable time step

function diffuse!(du, u, p,t)
    dijij = view(du, 2:Nx-1, 2:Ny-1,2:Nz-1)
    dij  = view(u, 2:Nx-1, 2:Ny-1,2:Nz-1)
    
    di1jx = view(u, 1:Nx-2, 2:Ny-1,2:Nz-1)
    dij1y = view(u, 2:Nx-1, 1:Ny-2, 2:Nz-1 )
    dij1z = view(u, 2:Nx-1, 2:Ny-1, 1:Nz-2)

    di2jx = view(u, 3:Nx  , 2:Ny-1,2:Nz-1  )
    dij2y = view(u, 2:Nx-1, 3:Ny,2:Nz-1  )                                                  # Stencil Computations
    dij2z = view(u, 2:Nx-1, 2:Ny-1,3:Nz)                                                  # Stencil Computations

    @. dijij = α  * (
        (di1jx - 2 * dij + di2jx)/Δx^2 +
        (dij1y - 2 * dij + dij2y)/Δy^2 +  
        (dij1z - 2 * dij + dij2z)/Δz^2   
        )                                               # Apply diffusion
### boundries
    @views @. du[1, :,:] += α  * (2*u[2, :,:] - 2*u[1, :,:])/Δx^2
    @views @. du[Nx, :,:] += α  * (2*u[Nx-1, :,:] - 2*u[Ny, :,:])/Δx^2
    
    @views @. du[:, 1, :] += α   * (2*u[:, 2,:]-2*u[:, 1,:])/Δy^2
    @views @. du[:, Ny, :] += α * (2*u[:, Nx-1,:]-2*u[:, Ny,:])/Δy^2                  # update boundary condition (Neumann BCs)

    @views @. du[:, 1, :] += α   * (2*u[:, :,2]-2*u[:, :,1])/Δz^2
    @views @. du[:, Ny, :] += α * (2*u[:, :,Nz-1]-2*u[:, :,Nz])/Δz^2    
  

end



#u_GPU= CUDA.zeros(Nx,Ny,Nz)
u_GPU= zeros(Float32,Nx,Ny,Nz)
du_GPU = similar(u_GPU)     
u_GPU[25:35, 25:35,25:35] .= 50f0;

diffuse!(du_GPU, u_GPU,0,0)

u_GPU[24,30,30]
du_GPU


using OrdinaryDiffEq

tspan = (0f0, 80.0)
prob = ODEProblem(diffuse!, u_GPU, tspan)
steadyProb=SteadyStateProblem(prob)

@time sol = solve(prob,ROCK2(), dt=Δt,save_everystep=false,save_start=false) 
#@time sol = solve(steadyProb, dt=Δt,save_everystep=false,save_start=false) 
ROCK2()
Tsit5()

res = sol(100.0)

maximum(res)

@time for i in 1:1000                                                                     # Apply the diffuse 1000 time to let the heat spread a long the rod      

    diffuse!(du_GPU, u_GPU,0,0)

    u_GPU = u_GPU + Δt * du_GPU

end;












                                                                            

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
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)
