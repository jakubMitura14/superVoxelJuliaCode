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
using KernelAbstractions
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern_unrolled.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")



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



"""
given the coordinates of the sampled points and the image cartesian coordinates we want to interpolate the values of the sampled points to the image
will iterate through sampled points and each time will check weather after cail or floor operation on the coordinates of the sampled point we get the same
image coordinate as the coordinate analysed by this thread in kernel next we calculate distance from this sampled point to the image coordinate 
and divide value of sampled point by the distance and add it to the image coordinate value; we also store the sum of weights for all sampled points
we want to associate with given point in the image so after iterating through sample points we will divide the value of the image coordinate by the sum of weights
"""
@kernel function inverse_interp_kern(@Const(image_cartesian), @Const(coords), @Const(len_coords), res)
    index = @index(Global)
    res[index] = 0.0
    # dist_sum=0.0
    shared_arr = @localmem Float32 (128, 4)
    shared_arr[@index(Local, Linear), 1] = 0.0
    shared_arr[@index(Local, Linear), 2] = 0.0
    shared_arr[@index(Local, Linear), 3] = 0.0
    shared_arr[@index(Local, Linear), 4] = 0.0


    for i in range(1, (len_coords - 1))
        if (image_cartesian[index, 1] == floor(coords[1, i]))
            shared_arr[@index(Local, Linear), 1] = floor(coords[1, i])
        end

        if (image_cartesian[index, 1] == ceil(coords[1, i]))
            shared_arr[@index(Local, Linear), 1] = ceil(coords[1, i])
        end

        if (image_cartesian[index, 2] == floor(coords[2, i]))
            shared_arr[@index(Local, Linear), 2] = floor(coords[1, 2])
        end

        if (image_cartesian[index, 2] == ceil(coords[2, i]))
            shared_arr[@index(Local, Linear), 2] = ceil(coords[2, i])
        end

        if (image_cartesian[index, 3] == floor(coords[3, i]))
            shared_arr[@index(Local, Linear), 3] = floor(coords[3, i])
        end

        if (image_cartesian[index, 3] == ceil(coords[3, i]))
            shared_arr[@index(Local, Linear), 3] = ceil(coords[3, i])
        end


        if ((shared_arr[@index(Local, Linear), 1] > 0) && (shared_arr[@index(Local, Linear), 2] > 0) && (shared_arr[@index(Local, Linear), 3] > 0))
            # dist= sqrt(( ((image_cartesian[index,1] -shared_arr[@index(Local, Linear),1] ) ^2)+((image_cartesian[index,2] -shared_arr[@index(Local, Linear),2] ) ^2)+((image_cartesian[index,3] -shared_arr[@index(Local, Linear),3] ) ^2) ))
            shared_arr[@index(Local, Linear), 4] += (((image_cartesian[index, 1] - coords[1, i])^2) + ((image_cartesian[index, 2] - coords[2, i])^2) + ((image_cartesian[index, 3] - coords[3, i])^2))
            # @print("   ** $(dist)   **")
            # @print("   **i $i $(@index(Local, Linear))   **")
            res[index] += (coords[4, i] * (((image_cartesian[index, 1] - coords[1, i])^2) + ((image_cartesian[index, 2] - coords[2, i])^2) + ((image_cartesian[index, 3] - coords[3, i])^2)))
        end
    end
    if (res[index] > 0.5)
        # @print("  $(res[index])  "  )
        res[index] = res[index] / shared_arr[@index(Local, Linear), 4]
    end

end#inverse_interp_kern

"""
given radiuss of the supervoxel and the size of the image we want to create (cube with edge lenth a) 
it will return 3 dimansional array that is result of inversly interpolated values of the sampled points
where each supervoxel has associated consecutive integer value, the shapes of supervoxels are random (based on random weights)
"""
function create_synth_image_for_test(radiuss,a)
    diam = radiuss * 2
    num_weights_per_point = 6
    image_shape = (a, a, a)

    print("\n uuuuuuuuuuuuuuuuuuuuuuu -2 \n")
    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
    print("\n uuuuuuuuuuuuuuuuuuuuuuu -1 \n")

    sv_centers, control_points, tetrs, dims = example_set_of_svs
    source_arr = rand(Float32, image_shape)
    num_base_samp_points, num_additional_samp_points = 3, 2


    #get the number of threads and blocks needed for the kernel
    threads_point_info, blocks_point_info, pad_point_info = prepare_for_kern(size(tetrs))
    max_index = size(tetrs)[1]
    num_base_samp_points, num_additional_samp_points = 3, 2
    #put on GPU
    tetr_dat = CuArray(Float32.(tetrs))
    sv_centers = CuArray(Float32.(sv_centers))
    control_points = CuArray(Float32.(control_points))
    source_arr = CuArray(Float32.(source_arr))
    tetr_dat_out = CuArray(Float32.(tetrs))

    out_sampled_points = CUDA.zeros((size(tetr_dat)[1], num_base_samp_points + (3 * num_additional_samp_points), 5))
    #initialize shadow memory
    dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
    weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
    weights = weights .- 0.50001
    weights = (weights) .* 100
    weights = CuArray(Float32.(tanh.(weights * 0.02)))

    print("\n uuuuuuuuuuuuuuuuuuuuuuu 000 \n")

    control_points_out = CuArray(Float32.(control_points))
    threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(size(control_points), size(weights))
    call_apply_weights_to_locs_kern(control_points, control_points_out, weights, radiuss, threads_apply_w, blocks_apply_w)

    control_points = control_points_out
    ### execute kernel no autodiff
    @cuda threads = threads_point_info blocks = blocks_point_info set_tetr_dat_kern_forward(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, max_index)
    tetr_dat = tetr_dat_out
    @cuda threads = threads_point_info blocks = blocks_point_info point_info_kern_forward(tetr_dat_out, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points, max_index)

    out_sampled_points = Array(out_sampled_points)
    print("\n uuuuuuuuuuuuuuuuuuuuuuu 111 \n")


    #getting coordinates of each sampled point and assigning them a value
    coords = map(index -> invert(splitdims(out_sampled_points[index, :, 3:5])), 1:size(out_sampled_points)[1])
    #for now we assign value that is just integer index of which supervoxel they are from we divide by 24 to get the supervoxel index not tetrahedron index
    coords = map(index -> map(el -> [el..., Int(ceil(index / 24))], coords[index]), 1:size(out_sampled_points)[1])
    coords = reduce(vcat, coords)
    coords = reduce(hcat, coords) #shape 4xn
    #now we get cartesian coordinates of the image 
    image_cartesian = get_base_indicies_arr(image_shape)
    sh = size(image_cartesian)
    image_cartesian = reshape(image_cartesian, (sh[1] * sh[2] * sh[3], 3))



    print("\n uuuuuuuuuuuuuuuuuuuuuuu 222 \n")

    # coords=coords[:,1:20]
    # image_cartesian=image_cartesian[1:9000,:] #
    len_coords = size(coords)[2]
    len_image_coords = size(image_cartesian)[1]
    res = CUDA.zeros(len_image_coords)
    dev = get_backend(res)
    inverse_interp_kern(dev, 128)(CuArray(image_cartesian), CuArray(coords), len_coords, res, ndrange=(len_image_coords))
    KernelAbstractions.synchronize(dev)

    res_im=reshape(Array(res), (a,a,a))
    return res_im
end

radiuss = Float32(4.0)
a = 256
print("\n iiii \n")  #1536
# create_synth_image_for_test(radiuss,a)


dims=(4,4,4)
flattened_triangles_a=get_flattened_triangle_data(dims) 
flattened_triangles_b=get_flattened_triangle_data_faster(dims) 
flattened_triangles_a==flattened_triangles_b
size(flattened_triangles_a)
a
# # using MedImages
# using PyCall
# sitk = pyimport_conda("SimpleITK", "simpleitk")
# # sitk.GetImageFromArray(res_im)
# sitk.WriteImage( sitk.GetImageFromArray(res_im),"/workspaces/superVoxelJuliaCode/superVoxelJuliaCode/data/res_im.nii.gz")
# a
# ##### viss


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

