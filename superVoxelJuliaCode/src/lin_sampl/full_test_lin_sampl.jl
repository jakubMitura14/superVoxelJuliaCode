"""
we generate a full test case on artifivcial data to test the linear sampling method
1) we get random supervoxel control points using random weights
2) we get a diffrent gaussian for each supervoxel keeping supervoxels that are close together having significantly diffrent kl divergence
3) we iterate over cartesian grid - check to which tetrahedron and by extension supervoxel this point belongs
    and we sample from corresponding gaussian
4) we visualize image using MedEye
5) we then run full optimazation loop using our algorithm and check weather points that we got at the begining and the
    ones that we got after the optimization are close to each other - we use in loss a minimazation of variance but our metric
    is a mean squared distance between points of sythethized gold standard and inferred points

"""

using Revise
using Meshes
using LinearAlgebra
using GLMakie
using Combinatorics
using SplitApplyCombine
using CUDA
using Combinatorics
using Random
using Statistics


includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")