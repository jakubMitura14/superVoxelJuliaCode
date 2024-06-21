using Revise, CUDA,HDF5
using Meshes
using LinearAlgebra
using GLMakie
using Combinatorics
using SplitApplyCombine
using CUDA
using Combinatorics
using Random
using Statistics
using ChainRulesCore
using Test
using ChainRulesTestUtils
using EnzymeTestUtils
using Logging, FiniteDifferences, FiniteDiff
using Interpolations
using KernelAbstractions,Dates
# using KernelGradients
using Zygote, Lux, LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using Lux, Random, Optimisers, Zygote
using LinearAlgebra

using Revise

includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/Lux_model.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point.jl")


"""
our loss function has 2 components that we would test separately one is variance of the borders 
    it should increase when variance of the image increases and vice versa
second is variance of supervoxels in case of the images from synth data this variance should be small 
    and basically quite similar to the variance related to image with constant value like ones
    it should increase a lot in case of adding noise with high variance - test multiplying and adding noise
"""