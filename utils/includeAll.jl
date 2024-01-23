using Revise
using CUDA, Enzyme, Test, Plots

includet("/workspaces/superVoxelJuliaCode/fullWithLoss/utils.jl")
includet("/workspaces/superVoxelJuliaCode/get_synth_data/generate_synth_simple.jl")
includet("/workspaces/superVoxelJuliaCode/fullWithLoss/createTestData.jl")
includet("/workspaces/superVoxelJuliaCode/fullWithLoss/getFeatures.jl")
includet("/workspaces/superVoxelJuliaCode/fullWithLoss/clusterKernels.jl")
includet("/workspaces/superVoxelJuliaCode/fullWithLoss/clusteringLoss.jl")
includet("/workspaces/superVoxelJuliaCode/fullWithLoss/getEdgesB.jl")
includet("/workspaces/superVoxelJuliaCode/fullWithLoss/applyGausses.jl")




