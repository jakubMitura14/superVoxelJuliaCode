using Revise, CUDA
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
using Interpolations,Dates
import CUDA
using KernelAbstractions
using LLVMLoopInfo

# ]add Revise, CUDA, Meshes, GLMakie, Combinatorics, SplitApplyCombine, ChainRulesCore, ChainRulesTestUtils, EnzymeTestUtils, Logging, FiniteDifferences, FiniteDiff, Interpolations, Dates, KernelAbstractions


includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")

includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern_unrolled.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_tetr.jl")

includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern _old.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")


function prepare_for_set_tetr_dat(tetr_dat_shape)
    # bytes_per_thread=6
    # blocks,threads,maxBlocks=computeBlocksFromOccupancy(point_info_kern,(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points), bytes_per_thread)
    # threads=256
    threads = 256

    # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
    # needed_blocks=ceil(total_num / threads_apply_w)
    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads)
    to_pad = (threads * needed_blocks) - tetr_dat_shape[1]

    return threads, needed_blocks, to_pad
end


radiuss = Float32(4.0)
diam = radiuss * 2
num_weights_per_point = 6
a = 36
image_shape = (a, a, a)

example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs
#here we get all tetrahedrons mapped to non modified locations
source_arr=rand(Float32, image_shape)
tetr_dat_out = zeros(size(tetrs))


# threads_point_info,blocks_point_info,pad_point_info=prepare_for_set_tetr_dat(size(tetrs))
tetr_dat=CuArray(Float32.(tetrs))
source_arr=CuArray(Float32.(source_arr))
control_points=CuArray(Float32.(control_points))
sv_centers=CuArray(Float32.(sv_centers))

tetr_dat_out = CUDA.zeros(size(tetr_dat)...)

# call_set_tetr_dat_kern(tetr_dat, source_arr, control_points, sv_centers)

function get_current_time()
    return Dates.now()
end
current_time = get_current_time()
a, a_pullback = rrule(call_set_tetr_dat_kern, tetr_dat, tetr_dat_out,source_arr, control_points, sv_centers);
a_pullback(tetr_dat)

println("Time taken (minutes): ", Dates.value(get_current_time() - current_time)/ 60000.0)

# r=get_current_time()
# a=(get_current_time()-r)
# Dates.value(a)/60000.0
# a/60000.0

# using Pkg
# Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")
# Pkg.add(url="https://github.com/JuliaGPU/KernelAbstractions.jl")


# ahead of the time compilation julia 11 https://docs.julialang.org/en/v1.11-dev/devdocs/aot/