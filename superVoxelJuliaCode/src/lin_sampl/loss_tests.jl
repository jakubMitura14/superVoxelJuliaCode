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
using ChainRulesCore


includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_tetr.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")


radiuss = Float32(4.0)
diam = radiuss * 2
num_weights_per_point = 6
a = 81
image_shape = (a, a, a)

example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs

dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
# control_points first dimension is lin_x, lin_y, lin_z, oblique
weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
# weights = rand(image_shape...)
weights = weights .- 0.50001
weights = (weights) .* 100
weights = Float32.(tanh.(weights * 0.02))


weights = ones(size(weights)...) .* 0.5#TODO remove this line

control_points_non_modified = copy(control_points)

control_points_size = size(control_points)

threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(control_points_size, size(weights))

control_points = call_apply_weights_to_locs_kern(CuArray(control_points), CUDA.zeros(size(control_points)...), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w)
# control_points=call_apply_weights_to_locs_kern(CuArray(control_points),CuArray(copy(control_points)),CuArray(weights),radiuss,threads_apply_w,blocks_apply_w)


control_points_non_modified[1, 1, 1, :, :, :]
control_points[1, 1, 1, :, :, :]


control_points_non_modified[2, 2, 2, :, :, :]
control_points[2, 2, 2, :, :, :]