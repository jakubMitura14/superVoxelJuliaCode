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
using Interpolations

includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern_unrolled.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern _old.jl")


"""
We want to create a synthethic image to detect the performance of the linear sampling kernel.
so we apply set of random weights 
"""

function prepare_for_kern(tetr_dat_shape)
    threads = 256

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
source_arr = rand(Float32, image_shape)
num_base_samp_points, num_additional_samp_points = 3, 2


#get the number of threads and blocks needed for the kernel
threads_point_info, blocks_point_info, pad_point_info = prepare_for_kern(size(tetrs))
max_index=size(tetrs)[1]
num_base_samp_points, num_additional_samp_points = 3, 2
#put on GPU
tetr_dat=CuArray(Float32.(tetrs))
sv_centers=CuArray(Float32.(sv_centers))
control_points=CuArray(Float32.(control_points))
source_arr=CuArray(Float32.(source_arr))
tetr_dat_out = CuArray(Float32.(tetrs))

out_sampled_points = CUDA.zeros((size(tetr_dat)[1], num_base_samp_points + (3 * num_additional_samp_points), 5))
#initialize shadow memory
dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
weights = weights .- 0.50001
weights = (weights) .* 100
weights = CuArray(Float32.(tanh.(weights * 0.02)))

control_points_out=CuArray(Float32.(control_points))
threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(size(control_points), size(weights))
call_apply_weights_to_locs_kern(control_points,control_points_out, weights, radiuss, threads_apply_w, blocks_apply_w)

control_points=control_points_out
### execute kernel no autodiff
@cuda threads = threads_point_info blocks = blocks_point_info set_tetr_dat_kern_forward(tetr_dat,tetr_dat_out,source_arr,control_points,sv_centers,max_index)
tetr_dat=tetr_dat_out
@cuda threads = threads_point_info blocks = blocks_point_info point_info_kern_forward(tetr_dat_out,out_sampled_points,source_arr,num_base_samp_points,num_additional_samp_points,max_index)

out_sampled_points=Array(out_sampled_points)
coords=map( index-> invert(splitdims(out_sampled_points[index,:,3:5])) ,1:size(out_sampled_points)[1])
coords = reduce(vcat, coords)

valss=ones(size(coords)[1])

# itp = interpolate(coords, valss, Gridded(Linear()))
sorted_coords = sort(coords)
sorted_coords = map(el->Float64.(el),sorted_coords)
sorted_coords = map(el->(el[1],el[2],el[3]) ,sorted_coords)

itp = linear_interpolation(sorted_coords, valss)
# itp = interpolate(sorted_coords,valss, Gridded(Linear()))

image = Array{Float64}(undef, a, a, a)
for i in 1:a, j in 1:a, k in 1:a
    image[i, j, k] = itp(i, j, k)
end


a



##### viss


# function fill_tetrahedron_data(tetr_dat, sv_centers, control_points, index)
#     center = map(axis -> sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), axis], [1, 2, 3])
#     corners = map(corner_num ->
#             map(axis -> control_points[Int(tetr_dat[index, corner_num, 1]), Int(tetr_dat[index, corner_num, 2]), Int(tetr_dat[index, corner_num, 3]), Int(tetr_dat[index, corner_num, 4]), axis], [1, 2, 3]), [2, 3, 4])
#     corners = [center, corners...]
#     return corners
# end

# function get_tetrahedrons_from_corners(corners)
#     # print("ttttttttttttt $(corners)")
#     # points = map(el -> Meshes.Point((el[1], el[2], el[3])), corners)
#     points = map(el -> (el[1], el[2], el[3]), corners)[1:4]
#     return Meshes.Tetrahedron(points...)
# end


# index=1
# # first_sv_tetrs= map(index->fill_tetrahedron_data(Array(tetr_dat_out), Array(sv_centers),Array(control_points),index),1:24)
# t=Array(tetr_dat_out)
# first_sv_tetrs=t[1:24,:,:][:,:,1:3]
# first_sv_tetrs= map(index->get_tetrahedrons_from_corners( invert(splitdims(first_sv_tetrs[index,:,:]))),1:24)

# viz(first_sv_tetrs, color=1:length(first_sv_tetrs))

