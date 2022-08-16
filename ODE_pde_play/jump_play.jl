using ModelingToolkit
using LinearAlgebra
# using OrdinaryDiffEq
# using DiffEqJump
using DiffEqSensitivity
using Zygote
using DifferentialEquations

p = [1]
rate1(u,p,t) = 0.5  # β*S*I

function affect1!(integrator)
  integrator.u[1] -= integrator.p[1]+1        # S -> S - 1
  integrator.u[2] += 1         # I -> I + 1
end
jump = ConstantRateJump(rate1,affect1!)
u₀    = [999,1,0]
tspan = (0.0,250.0)
prob = DiscreteProblem(u₀, tspan, p)
jump_prob = JumpProblem(prob, Direct(), jump)

sol = solve(jump_prob, SSAStepper())

function sum_of_solution(u₀,p)
    _prob = remake(prob,u0=u₀,p=p)
    sum(solve(_prob, SSAStepper()))
  end

du01,dp1 = Zygote.gradient(sum_of_solution,u₀,p)
