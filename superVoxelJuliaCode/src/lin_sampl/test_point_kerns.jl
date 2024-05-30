using Revise,CUDA
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
using Logging,FiniteDifferences,FiniteDiff
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")

# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_tetr.jl")

includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern _old.jl")


includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")


radiuss = Float32(4.0)
diam = radiuss * 2
num_weights_per_point = 6
a = 36
image_shape = (a, a, a)

example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs

dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
# control_points first dimension is lin_x, lin_y, lin_z, oblique
weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
# # weights = rand(image_shape...)
# weights = weights .- 0.50001
# weights = (weights) .* 100
# weights = Float32.(tanh.(weights * 0.02))



weights = ones(size(weights)...) .* 0.5

control_points_non_modified = copy(control_points)

control_points_size = size(control_points)

threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(control_points_size, size(weights))

# control_points = call_apply_weights_to_locs_kern(CuArray(control_points), CUDA.zeros(size(control_points)...), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w)
control_points=call_apply_weights_to_locs_kern(CuArray(control_points),CuArray(copy(control_points)),CuArray(weights),radiuss,threads_apply_w,blocks_apply_w)


"""
interpolation check - chek weather the value we got from interpolation make sense (we are meaking separete kernel just for this tests)
1) create arrays that will be used for interpolation check in such a way that we will create a separate 3d array for each axis and to the half of this axis it will have value 0 
   and the other half value 1 and then we will interpolate the value in the middle of the axis and check weather the value is 0.5
2) get arrays from above and mutate one zero close to queried point into 1 and check weather the value in the middle of the axis is above 0.5 but below 1
3) get arrays from above and mutate one one close to queried point into 0 and check weather the value in the middle of the axis is below 0.5 but above 1
4) get a array with various values (arrange consecutive integers) get a point arbitrary to one of the points and check 
    weather value is approximately equal to the value of the point
"""

function interpolation_kernel(d_data::CuDeviceArray{Float32, 3}, x::Float32, y::Float32, z::Float32, d_result::CuDeviceArray{Float32, 1})
    idx = threadIdx().x
    if idx == 1
        @threeDLinInterpol(d_data)
        d_result[1] = var2
    end
    return
end

function trilinear_interpolation(data::Array{Float32, 3}, x::Float32, y::Float32, z::Float32)
    d_data = CuArray(data)
    d_result = CUDA.zeros(Float32, 1)
    @cuda threads=1 interpolation_kernel(d_data, x, y, z, d_result)
    return Array(d_result)[1]
end


@testset "Interpolation Tests" begin
    # Test 1
    for axis in 1:3
        data = ones(10, 10, 10)
        data[1:5, :, :] .= 0
        @test trilinear_interpolation(5.5, 5.5, 5.5, data) ≈ 0.5
    end

    # Test 2
    for axis in 1:3
        data = ones(10, 10, 10)
        data[1:5, :, :] .= 0
        data[5, 5, 5] = 1
        @test trilinear_interpolation(5.5, 5.5, 5.5, data) > 0.5
        @test trilinear_interpolation(5.5, 5.5, 5.5, data) < 1
    end

    # Test 3
    for axis in 1:3
        data = ones(10, 10, 10)
        data[1:5, :, :] .= 0
        data[6, 6, 6] = 0
        @test trilinear_interpolation(5.5, 5.5, 5.5, data) < 0.5
        @test trilinear_interpolation(5.5, 5.5, 5.5, data) > 0
    end

    # Test 4
    data = reshape(1:1000, 10, 10, 10)
    @test trilinear_interpolation(5.5, 5.5, 5.5, data) ≈ data[5, 5, 5]
end




"""
variance check - check weather the variance of the values that we get from interpolation make sense (we are meaking separete kernel just for this tests)
1) get array of constant values and check weather the variance is 0
2) mutate array from 1 and set single voxel close to point at 1 and check weather the variance has increased above 0
3) mutete second voxel with value 2 and check weather the variance has increased more then in case of 2 ...
   do this with all surrounding voxels of the point and be sure that the values are increaing with each addition
4) get array for each axis where first half of the axis is random and second is ones and check if the point a bit closer to random part has higher variance than a point 
    a bit closer to ones part
   """
function variance_check_kernel(d_data::CuDeviceArray{Float32, 3}, x::Float32, y::Float32, z::Float32, d_result::CuDeviceArray{Float32, 1})
    idx = threadIdx().x
    if idx == 1
        d_result[1] = @threeD_loc_var(d_data, x, y, z)
    end
    return
end

function variance_check(data::Array{Float32, 3}, x::Float32, y::Float32, z::Float32)
    d_data = CuArray(data)
    d_result = CUDA.zeros(Float32, 1)
    @cuda threads=1 variance_check_kernel(d_data, x, y, z, d_result)
    return Array(d_result)[1]
end

@testset "Variance Tests" begin
    # Test 1
    data = ones(10, 10, 10)
    @test variance_check(data, 5.5, 5.5, 5.5) == 0

    # Test 2
    data[5, 5, 5] = 2
    var1 = variance_check(data, 5.5, 5.5, 5.5)
    @test var1 > 0

    # Test 3
    data[6, 6, 6] = 3
    var2 = variance_check(data, 5.5, 5.5, 5.5)
    @test var2 > var1

    # Test 4
    data = ones(10, 10, 10)
    data[1:5, :, :] = rand(5, 10, 10)
    var1 = variance_check(data, 4.5, 5.5, 5.5)
    var2 = variance_check(data, 6.5, 5.5, 5.5)
    @test var1 > var2
end

"""
tetr dat check - we are testing the set_tetr_dat_kern kernel - tetr dat was updated correctly so 
   a) first isolate inormation about some tetrahedron its indicies and their location in unmodified StatsBase
   b) then check that the location of the tetrahedron was updated correctly
   c) check weater the value that got associated with tetrahedron point makes sense (using interpolations tests)
   d) establish if centroid of the tetrahedron  base is in the middle of the points of a tetrahedron base
   e) check weather centroid interpolated correctly
   f) check weather sv center interpolated correctly
"""
# Test data initialization
source_arr = rand(10, 10, 10)
control_points = rand(10, 10, 10, 3)
sv_centers = rand(10, 10, 10, 3)
tetr_dat = rand(10, 5, 4)
tetr_dat_out = zeros(size(tetr_dat))

# Invoke the kernel
@cuda threads=256 blocks=10 set_tetr_dat_kern(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers)

# Test functions
@testset "set_tetr_dat_kern tests" begin
    @testset "tetrahedron location update" begin
        # Get indices and locations from the original tetr_dat
        indices = [i for i in 1:size(tetr_dat, 1)]
        locations = [tetr_dat[i, :, :] for i in indices]
        for (index, location) in zip(indices, locations)
            @test tetr_dat_out[index, :, :] == location
        end
    end

    @testset "interpolation tests" begin
        # Check if the interpolated value is close to the original value
        for i in 1:size(tetr_dat_out, 1)
            @test isapprox(tetr_dat_out[i, :, 4], source_arr[i, :, :]; atol=1e-5)
        end
    end

    @testset "centroid tests" begin
        # Calculate the centroid of the tetrahedron base
        for i in 1:size(tetr_dat_out, 1)
            centroid = sum(tetr_dat_out[i, 2:4, 1:3], dims=1) / 3
            @test tetr_dat_out[i, 5, 1:3] == centroid
        end
    end

    @testset "sv center tests" begin
        for i in 1:size(tetr_dat_out, 1)
            @test tetr_dat_out[i, 1, 1:3] == sv_centers[i, :, :]
        end
    end
end

"""
testing point_info_kern function 
check base sample points
1) check weather the base sample points are on the line between the center of the triangle and the center of the super voxel
2) check weather the interpolation value of the sampled points agrees with interpolations checks
3) check is weight associated with sample point is proportional to the distance between them
check additional sample points
4) check weather they are on the line between the last base sample point and corners of the base of the tetrahedron
5) check weather the interpolation value of the sampled points agrees with interpolations checks
6) check is weight associated with sample point is proportional to the distance between them
"""


# Test data initialization
source_arr = rand(10, 10, 10)
control_points = rand(10, 10, 10, 3)
sv_centers = rand(10, 10, 10, 3)
tetr_dat = rand(10, 5, 4)
tetr_dat_out = zeros(size(tetr_dat))
base_sample_points = rand(10, 10, 10, 3)
additional_sample_points = rand(10, 10, 10, 3)

# Invoke the kernel
@cuda threads=256 blocks=10 point_info_kern(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, base_sample_points, additional_sample_points)

# Test functions
@testset "point_info_kern tests" begin
    @testset "base sample points tests" begin
        # 1) Check if the base sample points are on the line between the center of the triangle and the center of the super voxel
        @test all(isapprox.(base_sample_points, sv_centers, atol=1e-5))
        # 2) Check if the interpolated value of the sampled points agrees with interpolations checks
        @test all(isapprox.(base_sample_points, control_points, atol=1e-5))
        # 3) Check if weight associated with sample point is proportional to the distance between them
        @test all(isapprox.(base_sample_points, tetr_dat_out, atol=1e-5))
    end
    @testset "additional sample points tests" begin
        # 4) Check if they are on the line between the last base sample point and corners of the base of the tetrahedron
        @test all(isapprox.(additional_sample_points, base_sample_points[end, :, :, :], atol=1e-5))
        # 5) Check if the interpolation value of the sampled points agrees with interpolations checks
        @test all(isapprox.(additional_sample_points, control_points, atol=1e-5))
        # 6) Check if weight associated with sample point is proportional to the distance between them
        @test all(isapprox.(additional_sample_points, tetr_dat_out, atol=1e-5))
    end
end


"""
visualization
visualize the points with weights as balls and the line between the center of the triangle and the center of the super voxel plus lines and balls for base and additional sample points
"""
