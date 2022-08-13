using Lux, Random, Optimisers, Zygote

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

# Construct the layer
model = Chain(BatchNorm(128), Dense(128, 256, tanh), BatchNorm(256),
              Chain(Dense(256, 1, tanh), Dense(1, 10)))


import SimpleContainerGenerator

mkpath("juliaCudefromSimple")
cd("juliaCudefromSimple")

pkgs = [
    "Lux", # Replace Foo, Bar, Baz, etc. with the names of actual packages that you want to use
    "Optimisers",
    "Zygote",
    "HDF5",
    "BSON",
    "Distributions",
    "Clustering",
    "ParallelStencil",
    "CUDA",
    "Hyperopt",
    "PythonCall",
    "Enzyme",
    "DifferentialEquations",
    "DataDrivenDiffEq",
    "DiffEqParamEstim",
    "Turing",
    "ModelingToolkit",
    "NBodySimulator",
    "StaticArrays",
    "DiffEqUncertainty",
    "MCMCChains",
    "KernelDensity",
    "Cuba",
    "DiffEqGPU",
    "Meshes",
    "MeshViz",
    "GLMakie",
    "Flux3D",
    "Manifolds",
    "ManifoldsBase",
    "Manopt",
    "Optim",
    "JuMP",
    "JunctionTrees",
    "HiGHS",
    "InferOpt",
    "Flux",
    "Setfield",
    "Kaleido",
    "Match",
    "MedEye3d",
    "Plots",
    "MLUtils",
    "DataAugmentation",
    "Graphs",
    "GridGraphs",
    "LinearAlgebra",
    "ProgressMeter",
    "UnicodePlots"
  

]
julia_version = v"1.7.1"

SimpleContainerGenerator.create_dockerfile(pkgs;
                                            julia_version = julia_version,
                                            output_directory = pwd())