"""
main idea is to check weather a line that goes from sv center to control point is inside the supervoxel
In order to test it we need to define a mesh which surface is defined by the set of control points

Plan is to define in meshes.jl the shape using a set of tetrahydras - we then need to check weather they intersect (they should not)
we can then later also test sampling scheme and check if a sampled point is inside any of the tetrahedron 
meshes additionally supply the visualization functionalities

!! important we assume that weights are in the range between -1 and 1 (so basically after tanh)
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


includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern.jl")
includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")


dims=(7,7,7)
dims_plus=(dims[1]+1,dims[2]+1,dims[3]+1)
radiuss=3.0
diam=radiuss*2
num_weights_per_point=6
example_set_of_svs=initialize_centers_and_control_points(dims,radiuss)
sv_centers,control_points,tetrs=example_set_of_svs   # ,lin_x_add,lin_y_add,lin_z_add

# control_points first dimension is lin_x, lin_y, lin_z, oblique
# weights=zeros((dims_plus[1],dims_plus[2],dims_plus[3],num_weights_per_point))
weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
weights=weights.-0.50001
weights=(weights).*100
weights = tanh.(weights*0.02)



threads=(2,2,2)
blocks=(2,2,2)
# TODO() calculate needed number of threads and blocks and add padding if needed
# https://cuda.juliagpu.org/stable/lib/driver/#Occupancy-API
# kernel = @cuda launch=false getBlockTpFpFn(args...) 
# threads ,blocks  = launch_configuration(apply_weights_to_locs_kern)

control_points_size=size(control_points)

# curr_x=7
# curr_y=3
# curr_z=1

# lin=(control_points_size[1]*control_points_size[2])*(curr_z-1) + (control_points_size[1])*(curr_y-1) + curr_x
# lin
# z = div(lin,(control_points_size[1]*control_points_size[2]))
# y = div(lin-(z*(control_points_size[1]*control_points_size[2])) , (control_points_size[1]))
# x = (lin-(z*(control_points_size[1]*control_points_size[2]))-(y*(control_points_size[1]) ) )

# x=x
# y=y+1
# z=z+1

# 65 % 64

"""
check the optimal launch configuration for the kernel
calculate the number of threads and blocks and how much padding to add if needed
"""
function prepare_for_apply_weights_to_locs_kern(control_points_shape,weights_shape)
    bytes_per_thread=0
    blocks_apply_w,threads_apply_w,maxBlocks=computeBlocksFromOccupancy(apply_weights_to_locs_kern,(CUDA.zeros(control_points_shape),CUDA.zeros(weights_shape),3,control_points_shape), bytes_per_thread)
    # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
    # needed_blocks=ceil(total_num / threads_apply_w)
    threads_res=(8,8,(floor(Int,threads_apply_w/64)))
    needed_blocks=(ceil(Int,control_points_shape[1]/threads_res[1]),ceil(Int,control_points_shape[2]/threads_res[2]),ceil(Int,control_points_shape[3]/threads_res[3]))

    return threads_apply_w,needed_blocks
end


threads_apply_w,blocks_apply_w=prepare_for_apply_weights_to_locs_kern(control_points_size,size(weights))

control_points=call_apply_weights_to_locs_kern(CuArray(control_points),CuArray(weights),radiuss,threads_apply_w,blocks_apply_w)
control_points=Array(control_points)



#how many main sample points we want to have between each triangle center and sv center in each tetrahedron
num_base_samp_points=3
num_additional_samp_points=2

varr=10.0
meann=-0.8

# source_arr = meann .+ sqrt(varr) .* randn(Int.(((dims.*(radiuss*2) ).+radiuss).+((radiuss*2)+1)) )
source_arr = meann .+ sqrt(varr) .* randn(Int.((dims.*(radiuss*2) ).+((radiuss*3))) )
out_sampled_points=zeros((size(tetrs)[1],num_base_samp_points+(3*num_additional_samp_points),5))


#we get the threads constant as it uses constant shared memory


tetrs=CuArray(tetrs)
out_sampled_points=CuArray(out_sampled_points)
source_arr=CuArray(source_arr)
control_points=control_points.+radiuss
sv_centers=sv_centers.+radiuss
control_points=CuArray(control_points)
sv_centers=CuArray(sv_centers)

size(source_arr)


bytes_per_thread=6

function prepare_for_point_info(tetr_dat_shape)
    bytes_per_thread=6
    #TODO (use dynamic shared memory below)
    # blocks,threads,maxBlocks=computeBlocksFromOccupancy(point_info_kern,(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points), bytes_per_thread)
    threads=256

    # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
    # needed_blocks=ceil(total_num / threads_apply_w)
    needed_blocks=ceil(Int,tetr_dat_shape[1]/threads)
    to_pad=(threads*needed_blocks)-tetr_dat_shape[1]

    return threads,needed_blocks,to_pad
end

threads_point_info,blocks_point_info,pad_point_info=prepare_for_point_info(size(tetrs))

8232/256

pad_point_info


out_sampled_points,tetr_dat=call_point_info_kern(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)

maximum(tetr_dat)
maximum(out_sampled_points)
tetr_dat[1,1,:]
source_arr[9,9,9]
maximum(source_arr)
tetrs[100,:,1]


maximum(tetr_dat[:,:,1])
maximum(tetr_dat[:,:,2])
maximum(tetr_dat[:,:,3])

control_points[1,1,1,:]

maximum(control_points[:,:,:,1])
maximum(control_points[:,:,:,2])
maximum(control_points[:,:,:,3])

size(out_sampled_points)

size(source_arr)


"""
now we want to visualize the points that were selected for sampling and their weights
    we will display their weights by the spheres of the radius equal to weight
"""

# points_mesh_a=Meshes.Point3.(invert(splitdims(out_sampled_points[:,3:5])))
# tt=invert(splitdims(tetr_dat))
# # tt=tt[2:4]
# points_mesh_b=Meshes.Point3.(tt)
# points_mesh=[points_mesh_a;points_mesh_b]

# spheres=[spheres;points_mesh]
# viz(spheres, color = 1:length(spheres),alpha=collect(1:length(spheres)).*0.9)
# # viz(points_mesh, color = 1:length(points_mesh))
# viz(tetrs, color = 1:length(tetrs))


"""
we want to check weather weighted sampling is working correctly by gettin base arr as the gaussian with known mean and variance
and checking weather our samples have similar weighted mean and variance
"""

using StatsBase

m1=mean(ssamp[:,1])
m2=sum(ssamp[:,1].*ssamp[:,2])/sum(ssamp[:,2])

m1,m2
v1=var(ssamp[:,1])
v2=sum(ssamp[:,2].*(ssamp[:,1].-m2).^2)/sum(ssamp[:,2])
v3=var(source_arr)
v1,v2,v3

# ,var(ssamp[:,1])