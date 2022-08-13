"""
You just semi-discretize via whatever (Gridap) to DiffEq, and then use the SteadyStateProblem which will have a backprop overload for the nonlinear solve
So on can specify the function that will tell about the value of given variable
    possibly this function can be just output of CNN maybe with some added interpolation ?

So first CNN - the one creating boundaries would not be the first layer - just the part of the OrdinaryDiffEq
        that will be the first layer ...

we can specify any function for the boundary conditions like we see in 
        https://docs.sciml.ai/dev/modules/MethodOfLines/boundary_conditions/

"""


#heterogenous pde example https://docs.sciml.ai/dev/modules/NeuralPDE/examples/heterogeneous/
# stochastic jump diffusion


#heat eq https://docs.sciml.ai/dev/modules/MethodOfLines/tutorials/heat/
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
using OrdinaryDiffEq,Plots
using ModelingToolkit, MethodOfLines, DomainSets, NonlinearSolve
#https://docs.sciml.ai/dev/modules/MethodOfLines/tutorials/heatss/
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0

bcs = [u(0, y) ~ x * y,
       u(1, y) ~ x * y,
       u(x, 0) ~ x * y,
       u(x, 1) ~ x * y]


# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
           y ∈ Interval(0.0, 1.0)]

@named pdesys = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])

dx = 0.1
dy = 0.1

# Note that we pass in `nothing` for the time variable `t` here since we
# are creating a stationary problem without a dependence on time, only space.
discretization = MOLFiniteDifference([x => dx, y => dy], nothing, approx_order=2)

prob = discretize(pdesys, discretization)
sol = NonlinearSolve.solve(prob, NewtonRaphson())

grid = get_discrete(pdesys, discretization)

u_sol = map(d -> sol[d], grid[u(x, y)])

using Plots

heatmap(grid[x], grid[y], u_sol, xlabel="x values", ylabel="y values",
        title="Steady State Heat Equation")



#Heat conduction system        
#https://docs.sciml.ai/dev/modules/ModelingToolkitStandardLibrary/tutorials/thermal_model/#Heat-Conduction-Model