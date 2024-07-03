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
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_variance_for_loss.jl")


"""
our loss function has 2 components that we would test separately one is variance of the borders 
    it should increase when variance of the image increases and vice versa
second is variance of supervoxels in case of the images from synth data this variance should be small 
    and basically quite similar to the variance related to image with constant value like ones
    it should increase a lot in case of adding noise with high variance - test multiplying and adding noise

Hovewer first checkwether we calculate per sv variance correctly    
"""

out_sampled_points=rand(Float32,24,9,5)
sizz_out=size(out_sampled_points)#(65856, 9, 5)
out_sampled_points_reshaped=reshape(out_sampled_points[:,:,1:2],(get_num_tetr_in_sv(),Int(round(sizz_out[1]/get_num_tetr_in_sv())),sizz_out[2],2))
size(out_sampled_points_reshaped)
out_sampled_points_reshaped=permutedims(out_sampled_points_reshaped,[2,1,3,4])
size(out_sampled_points_reshaped)
res=call_get_per_sv_variance(CuArray(out_sampled_points_reshaped))
res=Array(res)

values=reshape(out_sampled_points[1:24,:,1],24*9)
weights=reshape(out_sampled_points[1:24,:,2],24*9)

weighted_points=values.*weights
mean_weighted= sum(weighted_points)/sum(weights)
variance=sum(((values.-mean_weighted).^2).*weights)/sum(weights)



res[1]
mean_weighted[1]

a=round(res[1],digits=4)


b=round(variance[1],digits=4)

round(a/b,digits=1)