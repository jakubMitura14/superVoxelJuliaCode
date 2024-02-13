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
# control_points=call_apply_weights_to_locs_kern(CuArray(control_points),CuArray(weights),radiuss,threads,blocks)
# control_points=Array(control_points)



#how many main sample points we want to have between each triangle center and sv center in each tetrahedron
num_base_samp_points=3
num_additional_samp_points=2

varr=10.0
meann=-0.8

source_arr = meann .+ sqrt(varr) .* randn(Int.(dims.*(radiuss*2)))
out_sampled_points=zeros((size(tetrs)[1],num_base_samp_points+(3*num_additional_samp_points),5))
threads=(100,)
blocks=(2,)

tetrs=CuArray(tetrs)
out_sampled_points=CuArray(out_sampled_points)
source_arr=CuArray(source_arr)
control_points=CuArray(control_points)
sv_centers=CuArray(sv_centers)


call_point_info_kern(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads,blocks)


a
# """
# get a diffrence between coordinates in given axis of sv center and triangle center
# """
# macro get_diff_on_line_sv_tetr(coord_i,tetr_dat_coord,point_num)
#   return  esc(quote
#       ((tetr_dat[$tetr_dat_coord,$coord_i]-tetr_dat[1,$coord_i])*($point_num/(num_base_samp_points+1)))
# end)
# end



# macro my_ceil(x)
#   return  esc(quote
#   round($x+0.50001)
# end)
# end

# macro my_floor(x)
#   return  esc(quote
#   round($x-0.50001)
# end)
# end

# """
# calculate distance between set location and neighbouring voxel coordinates
# """
# macro get_dist(a,b,c)
#   return  esc(:(
#   (1/((((shared_arr[1]-round(shared_arr[1]+($a) ))^2+(shared_arr[2]-round(shared_arr[2] +($b)))^2+(shared_arr[3]-round(shared_arr[3]+($c)))^2)^2  )+0.00000001))#we add a small number to avoid division by 0
# ))
# end

# """
# get interpolated value at place
# """
# macro get_interpolated_val(source_arr,a,b,c)

#   return  esc(quote       
#       var3=(@get_dist($a,$b,$c))
#       var1+=var3
#       ($source_arr[Int(round(shared_arr[1]+($a))),Int(round(shared_arr[2]+($b))),Int(round(shared_arr[3]+($c)))]*(var3 ))
# end)
# end

# """
# used to get approximation of local variance
# """
# macro get_interpolated_diff(source_arr,a,b,c)

#   return  esc(quote       
#       ((($source_arr[Int(round(shared_arr[1]+($a))),Int(round(shared_arr[2]+($b))),Int(round(shared_arr[3]+($c)))])-var2)^2)*((@get_dist($a,$b,$c)) )
# end)
# end




# """
# simple kernel friendly interpolator - given float coordinates and source array will 
# 1) look for closest integers in all directions and calculate the euclidean distance to it 
# 2) calculate the weights for each of the 8 points in the cube around the pointadding more weight the closer the point is to integer coordinate
# """  
# macro threeDLinInterpol(source_arr)
#   ## first we get the total distances of all points to be able to normalize it later
#   return  esc(quote
#   var1=0.0#
 
#   var2=0.0
#   for a1 in -0.50001:0.50001:0.50001
#       for b1 in -0.50001:0.50001:0.50001
#           for c1 in -0.50001:0.50001:0.50001
#               var2+= @get_interpolated_val(source_arr,a1,b1,c1)
#           end
#       end
#   end        
#   var2=var2/var1
#   end)
# end

# """
# get local estimation of variance of the source array near the point
# """
# macro threeD_loc_var(source_arr)
#   ## first we get the total distances of all points to be able to normalize it later
#   return  esc(quote
#   @threeDLinInterpol(source_arr)
#   for a1 in -0.50001:0.50001:0.50001
#       for b1 in -0.50001:0.50001:0.50001
#           for c1 in -0.50001:0.50001:0.50001
#               var3+= @get_interpolated_diff(source_arr,a1,b1,c1)
#           end
#       end
#   end       
#   var2=var3/var1
#   end)
# end







# """
# each tetrahedron will have a set of sample points that are on the line between sv center and triangle center
# and additional one between corner of the triangle and the last main sample point 
# function is created to find ith sample point when max is is the number of main sample points (num_base_samp_points) +3 (for additional ones)
#   now we have in tetrs all of the triangles that create the outer skin of sv volume 
#   we need to no 
#   1. get a center of each triangle
#   2. draw a line between the center of the triangle and the center of the sv lets call it AB
#   3. divide AB into n sections (the bigger n the more sample points)
#   4. division between sections will be our main sample points morover we will get point in a middle between
#       last main zsample points and a verticies of main triangle that we got as input
#   5. we weight each point by getting the distance to the edges of the tetrahedron in which we are operating
#       the bigger the distance the bigger the importance of this sample point. 
#       a)in order to get this distance for main sample points we need to define lines between triangle verticies and the center of the sv
#       and then we need to project the sample point onto this line pluc the distance to previous and or next sample point
#       b)in case for additional ones - those that branch out from last main sample point we will approximate the spread by just getting the distance between
#       the last main sample point and the the vartex of the trangle that we used for it and using it as a diameter of the sphere which volume we will use for weighting those points
  
  
#   Implementation details:
#   probably the bst is to get all of the sample point per tetrahedron to be calculated in sequence over single thread
#   and then parallelize info on tetrahedrons 
#   In case of using float32 and reserving space for shadow memory for enzyme we probably can keep 2-3 floats32 in shared memory per thread
#   for x,y,z indicies of the supervoxels probably uint8 will be sufficient - morover we can unroll the loop
#   we will also try to use generic names of the variables to keep track of the register memory usage 
  
#   tetr_dat - array of 5 points indicies that first is sv center next 3 are verticies of the triangle that creates the base of the tetrahedron plus this triangle center we also have added point to indicate location in control_points so 5x4 array
#   num_base_samp_points - how many main sample points we want to have between each triangle center and sv center in each tetrahedron
#   shared_arr - array of length 3 where we have allocated space for temporarly storing the sample point that we are calculating
#   out_sampled_points - array for storing the value of the sampled point that we got from interpolation and its weight that depends on the proximity to the edges of the tetrahedron and to other points
#       hence the shape of the array is (num_base_samp_points+3)x5 where in second dimension first is the interpolated value of the point and second is its weight the last 3 entries are x,y,z coordinates of sampled point
#   source_arr - array with image on which we are working    
#   control_points - array of where we have the coordinates of the control points of the tetrahedron

#   num_additional_samp_points - how many additional sample points we want to have between the last main sample point and the verticies of the triangle that we used for it
# """
# function get_sample_point_num(tetr_dat,num_base_samp_points,shared_arr,out_sampled_points,source_arr,control_points,sv_centers,num_additional_samp_points)
#   #local variables are reused
#   var1=0.0
#   var2=0.0
#   var3=0.0

#   #TODO unroll the loops
#   #TODO try also calculating local directional variance of the source_arr (so iterate only over x y or z axis); local entrophy
#   #setting sv centers data
#   for axis in UInt8(1):UInt8(3)
#       shared_arr[axis]=sv_centers[Int(tetr_dat[1,1]),Int(tetr_dat[1,2]),Int(tetr_dat[1,3]),axis]
#   end#for axis
#   for axis in UInt8(1):UInt8(3)
#       tetr_dat[1,axis]=shared_arr[axis] #populate tetr dat wth coordinates of sv center
#   end#for axis
#   #performing interpolation result is in var2 and it get data from shared_arr
#   @threeDLinInterpol(source_arr)
#   #saving the result of sv center value
#   tetr_dat[1,4]=var2

#   for triangle_corner_num in UInt8(2):UInt8(4)
  
#       for axis in UInt8(1):UInt8(3)
#           shared_arr[axis]=control_points[Int(tetr_dat[triangle_corner_num,1]),Int(tetr_dat[triangle_corner_num,2]),Int(tetr_dat[triangle_corner_num,3]),Int(tetr_dat[triangle_corner_num,4]),axis]
#       end#for axis
#       for axis in UInt8(1):UInt8(3)
#           tetr_dat[triangle_corner_num,axis]=shared_arr[axis] #populate tetr dat wth coordinates of triangle corners
#       end#for axis
#       #performing interpolation result is in var2 and it get data from shared_arr
#       @threeD_loc_var(source_arr)
#       #saving the result of control point value
#       tetr_dat[triangle_corner_num,4]=var2
#   end#for triangle_corner_num

#   #get the center of the triangle
#   for triangle_corner_num in UInt8(2):UInt8(3)#we do not not need to loop over last triangle as it is already in the shared memory
#       #we add up x,y,z info of all triangles and in the end we divide by 3 to get the center
#       for axis in UInt8(1):UInt8(3)
#           shared_arr[axis]+=tetr_dat[triangle_corner_num,axis] 
#       end#for axis
#   end#for triangle_corner_num
#   for axis in UInt8(1):UInt8(3)
#       tetr_dat[5,axis]=(shared_arr[axis]/3)
#   end#for axis
#   @threeD_loc_var(source_arr)
#   #saving the result of centroid value
#   tetr_dat[5,4]=var2

#   var1=0.0
#   var2=0.0
#   #we iterate over rest of points in main sample points
#   for point_num in UInt8(1):UInt8(num_base_samp_points)

#       #we get the diffrence between the sv center and the triangle center
#       shared_arr[1]= @get_diff_on_line_sv_tetr(1,5,point_num)
#       shared_arr[2]=@get_diff_on_line_sv_tetr(2,5,point_num)
#       shared_arr[3]=@get_diff_on_line_sv_tetr(3,5,point_num)

#       ##calculate weight of the point
#       #first distance from next and previous point on the line between sv center and triangle center
#       var1=sqrt((shared_arr[1]/point_num)^2 +(shared_arr[2]/point_num)^2+(shared_arr[3]/point_num)^2)*2 #distance between main sample points (two times for distance to previous and next)
#       #now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
#       # we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
#       for triangle_corner_num in UInt8(1):UInt8(3)
#           #distance to the line between sv center and the  corner
#           var1+=sqrt((shared_arr[1] -@get_diff_on_line_sv_tetr(1,triangle_corner_num+1,point_num) )^2
#                          +(shared_arr[2] -@get_diff_on_line_sv_tetr(2,triangle_corner_num+1,point_num) )^2     
#                          +(shared_arr[3] -@get_diff_on_line_sv_tetr(3,triangle_corner_num+1,point_num) )^2     
#           ) 
#       end#for triangle_corner_num     

#       #now as we had looked into distance to other points in 5 directions we divide by 5 and save it to the out_sampled_points
#       out_sampled_points[point_num,2]=var1/5

    
#       ##time to get value by interpolation and save it to the out_sampled_points
#       #now we get the location of sample point
#       shared_arr[1]= tetr_dat[1,1]+shared_arr[1]
#       shared_arr[2]= tetr_dat[1,2]+shared_arr[2]
#       shared_arr[3]= tetr_dat[1,3]+shared_arr[3]
#       #performing interpolation result is in var2 and it get data from shared_arr
#       @threeDLinInterpol(source_arr)
#       #saving the result of interpolated value to the out_sampled_points
#       out_sampled_points[point_num,1]=var2
#       #saving sample points coordinates
#       out_sampled_points[point_num,3]=shared_arr[1]
#       out_sampled_points[point_num,4]=shared_arr[2]
#       out_sampled_points[point_num,5]=shared_arr[3]


#   end#for num_base_samp_points

#   ##### now we need to calculate the additional sample points that are branching out from the last main sample point
#   for n_add_samp in UInt8(1):UInt8(num_additional_samp_points)
#       for triangle_corner_num in UInt8(1):UInt8(3)

#           #now we need to get diffrence between the last main sample point and the triangle corner
#           shared_arr[1]=(tetr_dat[triangle_corner_num+1,1]-out_sampled_points[num_base_samp_points,3])*(n_add_samp/(num_additional_samp_points+1))
#           shared_arr[2]=(tetr_dat[triangle_corner_num+1,2]-out_sampled_points[num_base_samp_points,4])*(n_add_samp/(num_additional_samp_points+1))
#           shared_arr[3]=(tetr_dat[triangle_corner_num+1,3]-out_sampled_points[num_base_samp_points,5])*(n_add_samp/(num_additional_samp_points+1))


#           out_sampled_points[(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,2]=sqrt( shared_arr[1]^2+shared_arr[2]^2+shared_arr[3]^2)
#           ##time to get value by interpolation and save it to the out_sampled_points
#           #now we get the location of sample point
#           shared_arr[1]= out_sampled_points[num_base_samp_points,3]+shared_arr[1]
#           shared_arr[2]= out_sampled_points[num_base_samp_points,4]+shared_arr[2]
#           shared_arr[3]= out_sampled_points[num_base_samp_points,5]+shared_arr[3]

#           #performing interpolation result is in var2 and it get data from shared_arr
#           @threeDLinInterpol(source_arr)
#           #saving the result of interpolated value to the out_sampled_points
#           out_sampled_points[(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,1]=var2
#           #saving sample points coordinates
#           out_sampled_points[(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,3]=shared_arr[1]
#           out_sampled_points[(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,4]=shared_arr[2]
#           out_sampled_points[(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,5]=shared_arr[3]



#       end #for triangle_corner_num
#   end #for n_add_samp
#   return nothing
# end




# function get_sample_of_ind(tetr_dat_in)

#     shared_arr=[0.0,0.0,0.0]
#     out_sampled_points=zeros((num_base_samp_points+(3*num_additional_samp_points),5))
#     # source_arr=ones(Int.(dims.*(radiuss*2)))
#     get_sample_point_num(tetr_dat_in,num_base_samp_points,shared_arr,out_sampled_points,source_arr,control_points,sv_centers,num_additional_samp_points)
#     return out_sampled_points

# end #get_sample_of_ind
# curr_tetr=tetrs[1,:,:]
# outt=get_sample_of_ind(curr_tetr)



# source_arr[5,5,5]
# outt
# curr_tetr
# source_arr[3,3,3]

# curr_tetr

# a



# function testinterpol(source_arr,shared_arr)
#     var1=0.0
#     var2=0.0
#     var3=0.0
#     @threeD_loc_var(source_arr)
#     # return var2,source_arr[Int(round(shared_arr[1]+0.50001)),Int(round(shared_arr[2]+0.50001)),Int(round(shared_arr[3]+0.50001))]
#     return round(var2;digits=2),round(source_arr[Int(round(shared_arr[1])),Int(round(shared_arr[2])),Int(round(shared_arr[3]))];digits=2)
# end    

# # source_arr=ones((7,7,7))
# source_arr[3,3,3]=10
# source_arr[4,3,3]=20
# a,b=testinterpol(source_arr,[3.41,3.41,3.41])
# a,b=testinterpol(source_arr,[3.9,3.0,3.0])
# a,b=testinterpol(source_arr,[3.8,3.0,3.0])
# a,b=testinterpol(source_arr,[3.7,3.0,3.0])
# a,b=testinterpol(source_arr,[3.6,3.0,3.0])
# a,b=testinterpol(source_arr,[3.5,3.0,3.0])
# a,b=testinterpol(source_arr,[3.4,3.0,3.0])
# a,b=testinterpol(source_arr,[3.3,3.0,3.0])
# a,b=testinterpol(source_arr,[3.2,3.0,3.0])
# a,b=testinterpol(source_arr,[3.1,3.0,3.0])
# a,b=testinterpol(source_arr,[3.1,3.1,3.1])
# Int(round(a))

# sampleddd=map(indd->get_sample_of_ind(indd),1:length(tetrs))
# ssamp=vcat(sampleddd...)



# 1
# spheres=map(i->Meshes.Sphere((out_sampled_points[i,3],out_sampled_points[i,4],out_sampled_points[i,5]),out_sampled_points[i,2]/2),1:6)

# Meshes.Point3(tetr_dat[1,:])
# tetr_curr=Meshes.Tetrahedron(Meshes.Point3(tetr_dat[1,:]),Meshes.Point3(tetr_dat[2,:]),Meshes.Point3(tetr_dat[3,:]),Meshes.Point3(tetr_dat[4,:]))






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