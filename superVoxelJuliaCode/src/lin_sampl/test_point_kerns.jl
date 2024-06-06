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

# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
# includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_tetr.jl")

includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern _old.jl")


includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")


# radiuss = Float32(4.0)
# diam = radiuss * 2
# num_weights_per_point = 6
# a = 36
# image_shape = (a, a, a)

# example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
# sv_centers, control_points, tetrs, dims = example_set_of_svs

# dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
# # control_points first dimension is lin_x, lin_y, lin_z, oblique
# weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)


"""
interpolation check - chek weather the value we got from interpolation make sense (we are meaking separete kernel just for this tests)
!!!! currently we simplify and just do trilinear interpolation but points 1-4 will be usefull for testing more elaborate plans that will
    take sapcing into account
1) create arrays that will be used for interpolation check in such a way that we will create a separate 3d array for each axis and to the half of this axis it will have value 0 
   and the other half value 1 and then we will interpolate the value in the middle of the axis and check weather the value is 0.5
2) get arrays from above and mutate one zero close to queried point into 1 and check weather the value in the middle of the axis is above 0.5 but below 1
3) get arrays from above and mutate one one close to queried point into 0 and check weather the value in the middle of the axis is below 0.5 but above 1
4) get a array with various values (arrange consecutive integers) get a point arbitrary to one of the points and check 
    weather value is approximately equal to the value of the point
"""

function scale(itp::AbstractInterpolation{T,N,IT}, ranges::Vararg{AbstractRange,N}) where {T,N,IT}
    # overwriting this function becouse check_ranges giving error
    # check_ranges(itpflag(itp), axes(itp), ranges)
    ScaledInterpolation{T,N,typeof(itp),IT,typeof(ranges)}(itp, ranges)
end


function trilinear_interpolation_kernel_cpu(point, input_array)
    # index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
    c = (((
        input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] * (1 - (point[1] - floor(Int, point[1]))) +
        input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] * (point[1] - floor(Int, point[1]))
    )
          *
          (1 - (point[2] - floor(Int, point[2]))) +
          (input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
           +
           input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] * (point[1] - floor(Int, point[1])))
          *
          (point[2] - floor(Int, point[2])))
         *
         (1 - (point[3] - floor(Int, point[3])))
         +
         ((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
           +
           input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] * (point[1] - floor(Int, point[1])))
          *
          (1 - (point[2] - floor(Int, point[2]))) +
          (input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
           +
           input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] * (point[1] - floor(Int, point[1])))
          *
          (point[2] - floor(Int, point[2])))
         *
         (point[3] - floor(Int, point[3])))

    return c
end




"""
function that will be used for testing interpolation
"""
function interpolate_my(point, input_array, input_array_spacing)

    old_size = size(input_array)
    itp = interpolate(input_array, BSpline(Linear()))
    #we indicate on each axis the spacing from area we are samplingA
    A_x1 = 1:input_array_spacing[1]:(old_size[1])
    A_x2 = 1:input_array_spacing[2]:(old_size[2])
    A_x3 = 1:input_array_spacing[3]:(old_size[3])

    itp = extrapolate(itp, 0.0)
    itp = scale(itp, A_x1, A_x2, A_x3)
    return itp(point[1], point[2], point[3])
end#interpolate_my


### testing base cpu inmplementation of interpolation
input_array = rand(10, 10, 10)
point = [5.5, 5.5, 5.5]
input_array_spacing = [1.0, 1.0, 1.0]
@test interpolate_my(point, input_array, input_array_spacing) ≈ trilinear_interpolation_kernel_cpu(point, input_array)




"""
variance check - check weather the variance of the values that we get from interpolation make sense (we are meaking separete kernel just for this tests)
1) get array of constant values and check weather the variance is 0
2) mutate array from 1 and set single voxel close to point at 1 and check weather the variance has increased above 0
3) mutete second voxel with value 2 and check weather the variance has increased more then in case of 2 ...
   do this with all surrounding voxels of the point and be sure that the values are increaing with each addition
4) get array for each axis where first half of the axis is random and second is ones and check if the point a bit closer to random part has higher variance than a point 
    a bit closer to ones part
   """

function trilinear_variance_kernel_cpu(input_array, point)

    mean = (((
        input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])]
        *
        (1 - (point[1] - floor(Int, point[1]))) +
        input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])]
        *
        (point[1] - floor(Int, point[1]))
    )
             *
             (1 - (point[2] - floor(Int, point[2]))) +
             (input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])]
              *
              (1 - (point[1] - floor(Int, point[1])))
              +
              input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])]
              *
              (point[1] - floor(Int, point[1])))
             *
             (point[2] - floor(Int, point[2])))
            *
            (1 - (point[3] - floor(Int, point[3])))
            +
            ((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])]
              *
              (1 - (point[1] - floor(Int, point[1])))
              +
              input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])]
              *
              (point[1] - floor(Int, point[1])))
             *
             (1 - (point[2] - floor(Int, point[2]))) +
             (input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])]
              *
              (1 - (point[1] - floor(Int, point[1])))
              +
              input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])]
              *
              (point[1] - floor(Int, point[1])))
             *
             (point[2] - floor(Int, point[2])))
            *
            (point[3] - floor(Int, point[3])))
    ############ variance
    variance = (((
        ((input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] - mean)^2)
        *
        (1 - (point[1] - floor(Int, point[1]))) +
        ((input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] - mean)^2)
        *
        (point[1] - floor(Int, point[1]))
    )
                 *
                 (1 - (point[2] - floor(Int, point[2]))) +
                 (((input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] - mean)^2)
                  *
                  (1 - (point[1] - floor(Int, point[1])))
                  +
                  ((input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] - mean)^2)
                  *
                  (point[1] - floor(Int, point[1])))
                 *
                 (point[2] - floor(Int, point[2])))
                *
                (1 - (point[3] - floor(Int, point[3])))
                +
                ((((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] - mean)^2)
                  *
                  (1 - (point[1] - floor(Int, point[1])))
                  +
                  ((input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] - mean)^2)
                  *
                  (point[1] - floor(Int, point[1])))
                 *
                 (1 - (point[2] - floor(Int, point[2]))) +
                 (((input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] - mean)^2)
                  *
                  (1 - (point[1] - floor(Int, point[1])))
                  +
                  ((input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] - mean)^2)
                  *
                  (point[1] - floor(Int, point[1])))
                 *
                 (point[2] - floor(Int, point[2])))
                *
                (point[3] - floor(Int, point[3])))



    # d_result[1] = variance

    return variance
end



@testset "Variance Tests cpu " begin
    data = ones(10, 10, 10)
    @test trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) == 0

    data[5, 5, 5] = 2
    var1 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5))
    @test var1 > 0

    data[6, 5, 5] = 3
    var2 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5))
    @test var2 > var1

    data[5, 6, 5] = 4
    var3 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5))
    @test var3 > var2

    data[5, 5, 6] = 5
    var4 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5))
    @test var4 > var3

    data[6, 5, 6] = 6
    var5 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5))
    @test var5 > var4

    data[5, 6, 6] = 7
    var6 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5))
    @test var6 > var5

    data[6, 6, 5] = 8
    var7 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5))
    @test var7 > var6

    data[6, 6, 6] = 11
    var8 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5))
    @test var8 > var7


    data = ones(10, 10, 10)
    data[1:5, :, :] = rand(5, 10, 10)
    var_a = trilinear_variance_kernel_cpu(data, (5.1, 5.5, 5.5))
    var_b = trilinear_variance_kernel_cpu(data, (5.9, 5.5, 5.5))
    @test var_a > var_b

    data = ones(10, 10, 10)
    data[:, 1:5, :] = rand(10, 5, 10)
    var_a = trilinear_variance_kernel_cpu(data, (5.5, 5.1, 5.5))
    var_b = trilinear_variance_kernel_cpu(data, (5.5, 5.9, 5.5))
    @test var_a > var_b

    data = ones(10, 10, 10)
    data[:, :, 1:5] = rand(10, 10, 5)
    var_a = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.1))
    var_b = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.9))
    @test var_a > var_b



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






function fill_tetrahedron_data(tetr_dat, sv_centers, control_points, index)
    center = map(axis -> sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), axis], [1, 2, 3])
    corners = map(corner_num ->
            map(axis -> control_points[Int(tetr_dat[index, corner_num, 1]), Int(tetr_dat[index, corner_num, 2]), Int(tetr_dat[index, corner_num, 3]), Int(tetr_dat[index, corner_num, 4]), axis], [1, 2, 3]), [2, 3, 4])
    corners = [center, corners...]
    return corners
end

function get_tetrahedrons_from_corners(corners)
    points = map(el -> Meshes.Point((el[1], el[2], el[3])), corners)
    return Meshes.Tetrahedron(points...)
end



function prepare_for_set_tetr_dat(tetr_dat_shape)
    # bytes_per_thread=6
    # blocks,threads,maxBlocks=computeBlocksFromOccupancy(point_info_kern,(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points), bytes_per_thread)
    # threads=256
    threads = 128

    # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
    # needed_blocks=ceil(total_num / threads_apply_w)
    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads)
    to_pad = (threads * needed_blocks) - tetr_dat_shape[1]

    return threads, needed_blocks, to_pad
end



function call_set_tetr_dat_kern_test(tetr_dat, source_arr, control_points, sv_centers, threads, blocks, pad_point_info)

    tetr_shape = size(tetr_dat)
    to_pad_tetr = ones(pad_point_info, tetr_shape[2], tetr_shape[3]) * 2
    tetr_dat = vcat(tetrs, to_pad_tetr)
    tetr_dat_out = zeros(size(tetr_dat)...)

    tetr_dat = CuArray(Float32.(tetr_dat))
    source_arr = CuArray(Float32.(source_arr))
    control_points = CuArray(Float32.(control_points))
    sv_centers = CuArray(Float32.(sv_centers))
    tetr_dat_out = CuArray(Float32.(tetr_dat_out))

    # @cuda threads = threads blocks = blocks point_info_kern(CuStaticSharedArray(Float32, (128,3)),tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points)
    @cuda threads = threads blocks = blocks set_tetr_dat_kern_forward(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers)




    tetr_dat_out = tetr_dat_out[1:tetr_shape[1], :, :]

    # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
    return tetr_dat_out
end


# @testset "set_tetr_dat_kern tests" begin

#     radiuss = Float32(4.0)
#     diam = radiuss * 2
#     num_weights_per_point = 6
#     a = 36
#     image_shape = (a, a, a)

#     example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
#     sv_centers, control_points, tetrs, dims = example_set_of_svs
#     #here we get all tetrahedrons mapped to non modified locations
#     sv_tetrs= map(index->fill_tetrahedron_data(tetrs, sv_centers,control_points,index),1:(size(tetrs)[1]))
#     source_arr=rand(Float32, image_shape)
#     tetr_dat_out = zeros(size(tetrs))



#     threads_point_info,blocks_point_info,pad_point_info=prepare_for_set_tetr_dat(size(tetrs))
#     tetr_dat_out=call_set_tetr_dat_kern_test(tetrs,source_arr,control_points,sv_centers,threads_point_info,blocks_point_info,pad_point_info)


#     # sv_tetrs[1][1]
#     # tetr_dat_out[1,1,:][1:3]
#     # @testset "is tetr dat out populated correctly" begin
#         tetr_dat_out=Array(tetr_dat_out)
#         for v in eachindex(sv_tetrs)
#             sum_x=0.0
#             sum_y=0.0
#             sum_z=0.0
#             for p in eachindex(sv_tetrs[v])
#                 ## test is the location of the tetrahedron points was updated correctly
#                 @test sv_tetrs[v][p] == tetr_dat_out[v,p,:][1:3]
#                 ## check is interpolation of sv cenetr is correctly written
#                 if(p==1)
#                     @test tetr_dat_out[v,p,4] ≈trilinear_interpolation_kernel_cpu(sv_tetrs[v][p], source_arr)
#                 end
#                 ## check is variance of other points is correctly written   
#                 if(p>1)
#                     @test tetr_dat_out[v,p,4] ≈trilinear_variance_kernel_cpu(source_arr,sv_tetrs[v][p])
#                     sum_x+=sv_tetrs[v][p][1]
#                     sum_y+=sv_tetrs[v][p][2]
#                     sum_z+=sv_tetrs[v][p][3]
#                 end    
#             end
#             ## check is centroid of the tetrahedron base is in the middle of the points of a tetrahedron base    
#             @test tetr_dat_out[v,5,1] ≈(sum_x/3)
#             @test tetr_dat_out[v,5,2] ≈(sum_y/3)
#             @test tetr_dat_out[v,5,3] ≈(sum_z/3)
#             @test tetr_dat_out[v,5,4] ≈trilinear_variance_kernel_cpu(source_arr,((sum_x/3),(sum_y/3),(sum_z/3)))
#         end
# end


# Test functions

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

function prepare_for_point_info_kern(tetr_dat_shape)
    bytes_per_thread = 6
    #TODO (use dynamic shared memory below)
    # blocks,threads,maxBlocks=computeBlocksFromOccupancy(point_info_kern,(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points), bytes_per_thread)
    # threads=256
    threads = 128

    # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
    # needed_blocks=ceil(total_num / threads_apply_w)
    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads)
    to_pad = (threads * needed_blocks) - tetr_dat_shape[1]

    return threads, needed_blocks, to_pad
end





function call_point_info_kern_test(tetr_dat, source_arr, control_points, threads, blocks, pad_point_info, num_base_samp_points, num_additional_samp_points)

    tetr_shape = size(tetr_dat)
    to_pad_tetr = ones(pad_point_info, tetr_shape[2], tetr_shape[3]) * 2
    tetr_dat = vcat(tetr_dat, to_pad_tetr)

    tetr_dat = CuArray(Float32.(tetr_dat))
    source_arr = CuArray(Float32.(source_arr))
    control_points = CuArray(Float32.(control_points))
    out_sampled_points = CUDA.zeros((size(tetr_dat)[1], num_base_samp_points + (3 * num_additional_samp_points), 5))
    out_shape = size(out_sampled_points)
    to_pad_out = CUDA.ones(pad_point_info, out_shape[2], out_shape[3]) * 2
    out_sampled_points = vcat(out_sampled_points, to_pad_out)
    # @cuda threads = threads blocks = blocks point_info_kern(CuStaticSharedArray(Float32, (128,3)),tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points)
    @cuda threads = threads blocks = blocks point_info_kern_forward(tetr_dat, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points)
    out_sampled_points = out_sampled_points[1:out_shape[1], :, :]





    # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
    return out_sampled_points
end

# @testset "point_info_kern tests" begin

radiuss = Float32(4.0)
diam = radiuss * 2
num_weights_per_point = 6
a = 36
image_shape = (a, a, a)

example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs
#here we get all tetrahedrons mapped to non modified locations
sv_tetrs = map(index -> fill_tetrahedron_data(tetrs, sv_centers, control_points, index), 1:(size(tetrs)[1]))
source_arr = rand(Float32, image_shape)
num_base_samp_points, num_additional_samp_points = 3, 2

tetr_dat_out = zeros(size(tetrs))
threads_point_info, blocks_point_info, pad_point_info = prepare_for_set_tetr_dat(size(tetrs))
tetr_dat_out = call_set_tetr_dat_kern_test(tetrs, source_arr, control_points, sv_centers, threads_point_info, blocks_point_info, pad_point_info)


threads_point_info, blocks_point_info, pad_point_info = prepare_for_point_info_kern(size(tetrs))
out_sampled_points = call_point_info_kern_test(tetr_dat_out, source_arr, control_points, threads_point_info, blocks_point_info, pad_point_info, num_base_samp_points, num_additional_samp_points)


tetr_dat_out=Array(tetr_dat_out)


index=1
point_num=1
point_coords=( ((tetr_dat_out[index,5,1]-tetr_dat_out[index,1,1])*(point_num/(num_base_samp_points+1))),
                ((tetr_dat_out[index,5,2]-tetr_dat_out[index,1,2])*(point_num/(num_base_samp_points+1))),
                ((tetr_dat_out[index,5,3]-tetr_dat_out[index,1,3])*(point_num/(num_base_samp_points+1))))

out_sampled_points = Array(out_sampled_points)
out_sampled_points[1, :, :]
tetr_dat_out[1,:,:]



a
# sv_tetrs[1]
# Array(tetrs)[1,:,:]
# sv_centers[1,1,1,:]

# sv_tetrs[1][1]
# tetr_dat_out[1,1,:][1:3]
# @testset "is tetr dat out populated correctly" begin
# tetr_dat_out=Array(tetr_dat_out)
# for v in eachindex(sv_tetrs)
#     sum_x=0.0
#     sum_y=0.0
#     sum_z=0.0
#     for p in eachindex(sv_tetrs[v])
#         ## test is the location of the tetrahedron points was updated correctly
#         @test sv_tetrs[v][p] == tetr_dat_out[v,p,:][1:3]
#         ## check is interpolation of sv cenetr is correctly written
#         if(p==1)
#             @test tetr_dat_out[v,p,4] ≈trilinear_interpolation_kernel_cpu(sv_tetrs[v][p], source_arr)
#         end
#         ## check is variance of other points is correctly written   
#         if(p>1)
#             @test tetr_dat_out[v,p,4] ≈trilinear_variance_kernel_cpu(source_arr,sv_tetrs[v][p])
#             sum_x+=sv_tetrs[v][p][1]
#             sum_y+=sv_tetrs[v][p][2]
#             sum_z+=sv_tetrs[v][p][3]
#         end    
#     end
#     ## check is centroid of the tetrahedron base is in the middle of the points of a tetrahedron base    
#     @test tetr_dat_out[v,5,1] ≈(sum_x/3)
#     @test tetr_dat_out[v,5,2] ≈(sum_y/3)
#     @test tetr_dat_out[v,5,3] ≈(sum_z/3)
#     @test tetr_dat_out[v,5,4] ≈trilinear_variance_kernel_cpu(source_arr,((sum_x/3),(sum_y/3),(sum_z/3)))
# end


"""
visualization
visualize the points with weights as balls and the line between the center of the triangle and the center of the super voxel plus lines and balls for base and additional sample points
"""





#### getting data about first supervoxel (first 24 tetrahedrons in tetrs)


# function visualization()
#     radiuss = Float32(4.0)

#     a = 36
#     image_shape = (a, a, a)

#     example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
#     sv_centers, control_points, tetrs, dims = example_set_of_svs
#     size(tetrs)


#     first_sv_tetrs= map(index->fill_tetrahedron_data(tetrs, sv_centers,control_points,index),1:24)
#     first_sv_tetrs=map(get_tetrahedrons_from_corners,first_sv_tetrs)

#     viz(first_sv_tetrs, color=1:length(first_sv_tetrs))
# end




############################

# using Interpolations,Test
# function scale(itp::AbstractInterpolation{T,N,IT}, ranges::Vararg{AbstractRange,N}) where {T,N,IT}
#     # overwriting this function becouse check_ranges giving error
#     # check_ranges(itpflag(itp), axes(itp), ranges)
#     ScaledInterpolation{T,N,typeof(itp),IT,typeof(ranges)}(itp, ranges)
# end


# function interpolate_my(point, input_array, input_array_spacing)

#     old_size = size(input_array)
#     itp = interpolate(input_array, BSpline(Linear()))
#     #we indicate on each axis the spacing from area we are samplingA
#     A_x1 = 1:input_array_spacing[1]:(old_size[1])
#     A_x2 = 1:input_array_spacing[2]:(old_size[2])
#     A_x3 = 1:input_array_spacing[3]:(old_size[3])

#     itp = extrapolate(itp, 0.0)
#     itp = scale(itp, A_x1, A_x2, A_x3)
#     return itp(point[1], point[2], point[3])
# end#interpolate_my






# function trilinear_interpolation_kernel(point, input_array, input_array_spacing, d_result)

#     c = (((
#         input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] * (1 - (point[1] - floor(Int, point[1]))) +
#         input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] * (point[1] - floor(Int, point[1]))
#         )
#           *
#           (1 - (point[2] - floor(Int, point[2]))) +
#           (input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
#            +
#            input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] * (point[1] - floor(Int, point[1])))
#           *
#           (point[2] - floor(Int, point[2])))
#          *
#          (1 - (point[3] - floor(Int, point[3])))
#          +
#          ((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
#            +
#            input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] * (point[1] - floor(Int, point[1])))
#           *
#           (1 - (point[2] - floor(Int, point[2]))) +
#           (input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
#            +
#            input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] * (point[1] - floor(Int, point[1])))
#           *
#           (point[2] - floor(Int, point[2])))
#          *
#          (point[3] - floor(Int, point[3])))

#     d_result[1] = c

#     return c
# end

# function trilinear_variance_kernel_cpu(input_array,point)

#     mean = (((
#         input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])]
#          * (1 - (point[1] - floor(Int, point[1]))) +
#         input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])]
#          * (point[1] - floor(Int, point[1]))
#         )
#           *
#           (1 - (point[2] - floor(Int, point[2]))) +
#           (input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])]
#            * (1 - (point[1] - floor(Int, point[1])))
#            +
#            input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])]
#             * (point[1] - floor(Int, point[1])))
#           *
#           (point[2] - floor(Int, point[2])))
#          *
#          (1 - (point[3] - floor(Int, point[3])))
#          +
#          ((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])]
#           * (1 - (point[1] - floor(Int, point[1])))
#            +
#            input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])]
#             * (point[1] - floor(Int, point[1])))
#           *
#           (1 - (point[2] - floor(Int, point[2]))) +
#           (input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])]
#            * (1 - (point[1] - floor(Int, point[1])))
#            +
#            input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])]
#             * (point[1] - floor(Int, point[1])))
#           *
#           (point[2] - floor(Int, point[2])))
#          *
#          (point[3] - floor(Int, point[3])))
#     ############ variance
#     variance = (((
#         ((input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])]-mean)^2)
#          * (1 - (point[1] - floor(Int, point[1]))) +
#         ((input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])]-mean)^2)
#          * (point[1] - floor(Int, point[1]))
#         )
#           *
#           (1 - (point[2] - floor(Int, point[2]))) +
#           (((input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])]-mean)^2)
#            * (1 - (point[1] - floor(Int, point[1])))
#            +
#            ((input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])]-mean)^2)
#             * (point[1] - floor(Int, point[1])))
#           *
#           (point[2] - floor(Int, point[2])))
#          *
#          (1 - (point[3] - floor(Int, point[3])))
#          +
#          ((((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])]-mean)^2)
#           * (1 - (point[1] - floor(Int, point[1])))
#            +
#            ((input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])]-mean)^2)
#             * (point[1] - floor(Int, point[1])))
#           *
#           (1 - (point[2] - floor(Int, point[2]))) +
#           (((input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])]-mean)^2)
#            * (1 - (point[1] - floor(Int, point[1])))
#            +
#            ((input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])]-mean)^2)
#             * (point[1] - floor(Int, point[1])))
#           *
#           (point[2] - floor(Int, point[2])))
#          *
#          (point[3] - floor(Int, point[3])))



#     # d_result[1] = variance

#     return variance
# end



# @testset "Variance Tests cpu " begin
#     data = ones(10, 10, 10)
#     @test trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) == 0

#     data[5, 5, 5] = 2
#     var1 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) 
#     @test var1 > 0

#     data[6, 5, 5] = 3
#     var2 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) 
#     @test var2 > var1

#     data[5, 6, 5] = 4
#     var3 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) 
#     @test var3 > var2

#     data[5, 5, 6] = 5
#     var4 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) 
#     @test var4 > var3

#     data[6, 5, 6] = 6
#     var5 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) 
#     @test var5 > var4

#     data[5, 6, 6] = 7
#     var6 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) 
#     @test var6 > var5

#     data[6, 6, 5] = 8
#     var7 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) 
#     @test var7 > var6

#     data[6, 6, 6] = 11
#     var8 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5)) 
#     @test var8 > var7


#     data = ones(10, 10, 10)
#     data[1:5, :, :] = rand(5, 10, 10)
#     var_a= trilinear_variance_kernel_cpu(data, (5.1, 5.5, 5.5))
#     var_b= trilinear_variance_kernel_cpu(data, (5.9, 5.5, 5.5))
#     @test var_a > var_b

#     data = ones(10, 10, 10)
#     data[:, 1:5, :] = rand(10, 5, 10)
#     var_a= trilinear_variance_kernel_cpu(data, (5.5, 5.1, 5.5))
#     var_b= trilinear_variance_kernel_cpu(data, (5.5, 5.9, 5.5))
#     @test var_a > var_b

#     data = ones(10, 10, 10)
#     data[:, :, 1:5] = rand(10, 10, 5)
#     var_a= trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.1))
#     var_b= trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.9))
#     @test var_a > var_b



# end


# # ok so try to get a point quite close to the corner and check weater the influence of the corner opposite in given axis is getting smaller with increased relative spacing to other axes


# input_array = rand(10, 10, 10)


# input_array = reshape(collect(1:1000), (10, 10, 10))
# point = [5.5, 5.5, 5.5]
# input_array_spacing = [1.0, 1.0, 1.0]

# input_array[5, 5, 5]#445
# input_array[6, 5, 5]#446
# input_array[5, 6, 5]#455
# input_array[5, 5, 6]#545
# input_array[5, 6, 6]#555
# input_array[6, 6, 5]#456
# input_array[6, 5, 6]#546
# input_array[6, 6, 6]#556

# gold_res = interpolate_my(point, input_array, input_array_spacing)#324.59

# d_result = zeros(2)
# trilinear_interpolation_kernel(point, input_array, input_array_spacing, d_result)
# trilinear_variance_kernel(point, input_array, input_array_spacing, d_result)
# d_result[1]
# gold_res ≈ d_result[1]





# input_array = reshape(collect(1:1000), (10, 10, 10))
# point = [5.1, 5.6, 5.9]
# input_array_spacing = [1.0, 1.0, 1.0]

# input_array[5, 5, 5]#445
# input_array[6, 5, 5]#446
# input_array[5, 6, 5]#455
# input_array[5, 5, 6]#545
# input_array[5, 6, 6]#555
# input_array[6, 6, 5]#456
# input_array[6, 5, 6]#546
# input_array[6, 6, 6]#556


# trilinear_interpolation_kernel(point, input_array, input_array_spacing, d_result)

# # trilinear_interpolation_kernel_scaled(point, input_array, input_array_spacing, d_result)


# # input_array_spacing = [1.0, 1.0, 1.0]
# # point[1]=5.1
# # input_array_spacing[1]=2.0

# # a=point[1] - floor(Int, point[1])
# # rel_spac[1]=(input_array_spacing[1]/sum(input_array_spacing))*3
# # b=a+0.5
# # c=b^rel_spac[1]
# # f=c+0.5
# # # d=c0.5
# # # e=d+0.5
# # # f=c*rel_spac[1]
# # round(f;digits=2)

# # ((((((point[1] - floor(Int, point[1])) - 0.5)^2) / 0.5) + 0.5) * input_array_spacing[1])