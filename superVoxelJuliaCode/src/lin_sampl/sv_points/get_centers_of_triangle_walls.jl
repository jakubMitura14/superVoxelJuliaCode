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


includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")


function flip_num(base_ind,tupl,ind)
    arr=collect(tupl)
    if(arr[ind]==base_ind[ind])
        arr[ind]=base_ind[ind]+1
    else
        arr[ind]=base_ind[ind]
    end    
    return Tuple(arr)
end

"""
we can identify the line between two corners that go obliquely through the wall of the cube
it connects points that has 2 coordinates diffrent and one the same 
we can also find a point in the middle so it will be in lin_x if this common index is 1 and in lin_y if it is 2 and lin_z if 3
next if we have 1 it is pre and if 2 post
    control_points first dimension is lin_x, lin_y, lin_z, oblique
"""
function get_linear_between(base_ind,ind_1,ind_2)
    if(ind_1[1]==ind_2[1])
        return control_points[ind_1[1],base_ind[2],base_ind[3],1,:]
    end
    if(ind_1[2]==ind_2[2])
        return control_points[base_ind[1],ind_1[2],base_ind[3],2,:]
    end

    return control_points[base_ind[1],base_ind[2],ind_1[3],3,:]
end




function get_tetr_a(control_points,base_ind,corner)
    sv_center=sv_centers[base_ind[1],base_ind[2],base_ind[3],:]
    p_a=flip_num(base_ind,corner,1)
    p_b=flip_num(base_ind,corner,2)
    p_c=flip_num(base_ind,corner,3)


    p_ab=get_linear_between(base_ind,p_a,p_b)
    p_ac=get_linear_between(base_ind,p_a,p_c)
    p_bc=get_linear_between(base_ind,p_b,p_c)

    p_a=control_points[p_a[1],p_a[2],p_a[3],4,:]
    p_b=control_points[p_b[1],p_b[2],p_b[3],4,:]
    p_c=control_points[p_c[1],p_c[2],p_c[3],4,:]

    # p_ab=Meshes.Point3(p_ab)
    # p_ac=Meshes.Point3(p_ac)
    # p_bc=Meshes.Point3(p_bc)

    corner=control_points[corner[1],corner[2],corner[3],4,:]

    # print("coo corner $(corner) p_a $(p_a) p_b $(p_b) p_c $(p_c) \n ")

    return [(sv_center,corner,p_a,p_ab)
            , (sv_center,corner,p_ab,p_b)        
            , (sv_center,corner,p_b,p_bc)       
            , (sv_center,corner,p_bc,p_c)     
            , (sv_center,corner,p_a,p_ac)   
            , (sv_center,corner,p_ac,p_c)   
    ]
    # return [Meshes.Point3(get_triangle_center(corner,p_a,p_ab))
    #         ,Meshes.Point3(get_triangle_center(corner,p_ab,p_b))        
    #         ,Meshes.Point3(get_triangle_center(corner,p_b,p_bc))       
    #         ,Meshes.Point3(get_triangle_center(corner,p_bc,p_c))      
    #         ,Meshes.Point3(get_triangle_center(corner,p_a,p_ac))     
    #         ,Meshes.Point3(get_triangle_center(corner,p_ac,p_c))    
    # ]
    # return [p_a,p_b,p_c,p_ab,p_ac,p_bc,corner   ]
end




function get_tetrahedrons_of_sv(base_ind)
    return [
        get_tetr_a(control_points,base_ind,(base_ind[1],base_ind[2],base_ind[3]))
        ,get_tetr_a(control_points,base_ind,(base_ind[1]+1,base_ind[2]+1,base_ind[3]))
        ,get_tetr_a(control_points,base_ind,(base_ind[1],base_ind[2]+1,base_ind[3]+1))
        ,get_tetr_a(control_points,base_ind,(base_ind[1]+1,base_ind[2],base_ind[3]+1))
    ]
end #get_tetrahedrons_of_sv


####################################
"""
now we have in tetrs all of the triangles that create the outer skin of sv volume 
we need to no 
1. get a center of each triangle
2. draw a line between the center of the triangle and the center of the sv lets call it AB
3. divide AB into n sections (the bigger n the more sample points)
4. division between sections will be our main sample points morover we will get point in a middle between
    last main zsample points and a verticies of main triangle that we got as input
5. we weight each point by getting the distance to the edges of the tetrahedron in which we are operating
    the bigger the distance the bigger the importance of this sample point. 
    a)in order to get this distance for main sample points we need to define lines between triangle verticies and the center of the sv
    and then we need to project the sample point onto this line pluc the distance to previous and or next sample point
    b)in case for additional ones - those that branch out from last main sample point we will approximate the spread by just getting the distance between
    the last main sample point and the the vartex of the trangle that we used for it and using it as a diameter of the sphere which volume we will use for weighting those points


Implementation details:
probably the bst is to get all of the sample point per tetrahedron to be calculated in sequence over single thread
and then parallelize info on tetrahedrons 
In case of using float32 and reserving space for shadow memory for enzyme we probably can keep 2-3 floats32 in shared memory per thread
for x,y,z indicies of the supervoxels probably uint8 will be sufficient

"""





# #reshape for broadcast





function get_triangle_center(tr)
    center = (tr[2] + tr[3] + tr[4]) / 3
    return center
end



# """
# get a point between 2 other in 3D using weight how far from the first it should be
# if weight is 0 it is the same as point 1 if 1 it is the same as point 2
# """
# function  get_point_on_a_line(vertex_1,vertex_2,weight)
#     diff_x=vertex_2[1]-vertex_1[1]
#     diff_y=vertex_2[2]-vertex_1[2]
#     diff_z=vertex_2[3]-vertex_1[3]
#     return [vertex_1[1]+(diff_x*weight),vertex_1[2]+(diff_y*weight),vertex_1[3]+(diff_z*weight)]
# end



# """
# get a coordinate on a given acxis between sv center and some of the point stored in the tetr_dat
# we weight the distance between sv center and the point on the basis of variable i that indicates which main point it is
# """
# macro get_coordinate_on_line_sv_tetr(coord_i,tetr_dat_coord,point_num)
#     return  esc(quote
#         tetr_dat[1,$coord_i]+((tetr_dat[$tetr_dat_coord,$coord_i]-tetr_dat[1,$coord_i])*(point_num/(num_base_samp_points+1)))
#   end)
#   end
  
  """
  get a diffrence between coordinates in given axis of sv center and triangle center
  """
  macro get_diff_on_line_sv_tetr(coord_i,tetr_dat_coord,point_num)
    return  esc(quote
        ((tetr_dat[$tetr_dat_coord,$coord_i]-tetr_dat[1,$coord_i])*($point_num/(num_base_samp_points+1)))
  end)
  end

  """
  get a diffrence between coordinates in given axis of last normal sample point and triangle center
  """
  macro get_diff_on_line_last_tetr(coord_i,triangle_corner_num)
    return  esc(quote
    # (tetr_dat[$triangle_corner_num,$coord_i]-  
    # ((tetr_dat[5,$coord_i]-tetr_dat[1,$coord_i])*(num_base_samp_points/(num_base_samp_points+1))) #recomputing last sample point
    #  )*(0.5) 
    (((tetr_dat[5,$coord_i]-tetr_dat[1,$coord_i])*(num_base_samp_points/(num_base_samp_points+1))) #recomputing last sample point
     -tetr_dat[$triangle_corner_num,$coord_i])*(0.5) 
  
  end)
  end
  macro my_ceil(x)
    return  esc(quote
    round($x+0.5)
  end)
  end
  
  macro my_floor(x)
      return  esc(quote
      round($x-0.5)
    end)
    end


"""
simple kernel friendly interpolator - given float coordinates and source array will 
1) look for closest integers in all directions and calculate the euclidean distance to it 
2) calculate the weights for each of the 8 points in the cube around the pointadding more weight the closer the point is to integer coordinate
"""  
macro threeDLinInterpol(source_arr)
    ## first we get the total distances of all points to be able to normalize it later
    return  esc(quote
    var1=0.0    
    var1+=sqrt((shared_arr[1]-@my_floor(shared_arr[1]))^2+(shared_arr[2]-@my_floor(shared_arr[2]))^2+(shared_arr[3]-@my_floor(shared_arr[3]))^2)
    var1+=sqrt((shared_arr[1]-@my_ceil(shared_arr[1]))^2+(shared_arr[2]-@my_floor(shared_arr[2]))^2+(shared_arr[3]-@my_floor(shared_arr[3]))^2)
    var1+=sqrt((shared_arr[1]-@my_floor(shared_arr[1]))^2+(shared_arr[2]-@my_ceil(shared_arr[2]))^2+(shared_arr[3]-@my_floor(shared_arr[3]))^2)
    var1+=sqrt((shared_arr[1]-@my_floor(shared_arr[1]))^2+(shared_arr[2]-@my_floor(shared_arr[2]))^2+(shared_arr[3]-@my_ceil(shared_arr[3]))^2)
    var1+=sqrt((shared_arr[1]-@my_ceil(shared_arr[1]))^2+(shared_arr[2]-@my_ceil(shared_arr[2]))^2+(shared_arr[3]-@my_floor(shared_arr[3]))^2)
    var1+=sqrt((shared_arr[1]-@my_floor(shared_arr[1]))^2+(shared_arr[2]-@my_ceil(shared_arr[2]))^2+(shared_arr[3]-@my_ceil(shared_arr[3]))^2)
    var1+=sqrt((shared_arr[1]-@my_ceil(shared_arr[1]))^2+(shared_arr[2]-@my_floor(shared_arr[2]))^2+(shared_arr[3]-@my_ceil(shared_arr[3]))^2)
    var1+=sqrt((shared_arr[1]-@my_ceil(shared_arr[1]))^2+(shared_arr[2]-@my_ceil(shared_arr[2]))^2+(shared_arr[3]-@my_ceil(shared_arr[3]))^2)
    # ## now we get the final value by weightes summation
    var2= $source_arr[Int(@my_floor(shared_arr[1])),Int(@my_floor(shared_arr[2])),Int(@my_floor(shared_arr[3]))]*(sqrt((shared_arr[1]-@my_floor(shared_arr[1]))^2+(shared_arr[2]-@my_floor(shared_arr[2]))^2+(shared_arr[3]-@my_floor(shared_arr[3]))^2)/var1)
    var2+= $source_arr[Int(@my_ceil(shared_arr[1])) ,Int(@my_floor(shared_arr[2])),Int(@my_floor(shared_arr[3]))]*(sqrt((shared_arr[1]-@my_ceil(shared_arr[1]))^2+(shared_arr[2]-@my_floor(shared_arr[2]))^2+(shared_arr[3]-@my_floor(shared_arr[3]))^2)/var1)
    var2+= $source_arr[Int(@my_floor(shared_arr[1])),Int(@my_ceil(shared_arr[2])) ,Int(@my_floor(shared_arr[3]))]*(sqrt((shared_arr[1]-@my_floor(shared_arr[1]))^2+(shared_arr[2]-@my_ceil(shared_arr[2]))^2+(shared_arr[3]-@my_floor(shared_arr[3]))^2)/var1)
    var2+= $source_arr[Int(@my_floor(shared_arr[1])),Int(@my_floor(shared_arr[2])),Int(@my_ceil(shared_arr[3]))]*(sqrt((shared_arr[1]-@my_floor(shared_arr[1]))^2+(shared_arr[2]-@my_floor(shared_arr[2]))^2+(shared_arr[3]-@my_ceil(shared_arr[3]))^2)/var1)
    var2+= $source_arr[Int(@my_ceil(shared_arr[1])) ,Int(@my_ceil(shared_arr[2])) ,Int(@my_floor(shared_arr[3]))]*(sqrt((shared_arr[1]-@my_ceil(shared_arr[1]))^2+(shared_arr[2]-@my_ceil(shared_arr[2]))^2+(shared_arr[3]-@my_floor(shared_arr[3]))^2)/var1)
    var2+= $source_arr[Int(@my_floor(shared_arr[1])),Int(@my_ceil(shared_arr[2])) ,Int(@my_ceil(shared_arr[3]))]*(sqrt((shared_arr[1]-@my_floor(shared_arr[1]))^2+(shared_arr[2]-@my_ceil(shared_arr[2]))^2+(shared_arr[3]-@my_ceil(shared_arr[3]))^2)/var1)
    var2+= $source_arr[Int(@my_ceil(shared_arr[1])) ,Int(@my_floor(shared_arr[2])),Int(@my_ceil(shared_arr[3]))]*(sqrt((shared_arr[1]-@my_ceil(shared_arr[1]))^2+(shared_arr[2]-@my_floor(shared_arr[2]))^2+(shared_arr[3]-@my_ceil(shared_arr[3]))^2)/var1)
    var2+= $source_arr[Int(@my_ceil(shared_arr[1]))  ,Int(@my_ceil(shared_arr[2])) ,Int(@my_ceil(shared_arr[3]))]*(sqrt((shared_arr[1]-@my_ceil(shared_arr[1]))^2+(shared_arr[2]-@my_ceil(shared_arr[2]))^2+(shared_arr[3]-@my_ceil(shared_arr[3]))^2)/var1)
    
    end)



end




"""
each tetrahedron will have a set of sample points that are on the line between sv center and triangle center
and additional one between corner of the triangle and the last main sample point 
function is created to find ith sample point when max is is the number of main sample points (num_base_samp_points) +3 (for additional ones)
    now we have in tetrs all of the triangles that create the outer skin of sv volume 
    we need to no 
    1. get a center of each triangle
    2. draw a line between the center of the triangle and the center of the sv lets call it AB
    3. divide AB into n sections (the bigger n the more sample points)
    4. division between sections will be our main sample points morover we will get point in a middle between
        last main zsample points and a verticies of main triangle that we got as input
    5. we weight each point by getting the distance to the edges of the tetrahedron in which we are operating
        the bigger the distance the bigger the importance of this sample point. 
        a)in order to get this distance for main sample points we need to define lines between triangle verticies and the center of the sv
        and then we need to project the sample point onto this line pluc the distance to previous and or next sample point
        b)in case for additional ones - those that branch out from last main sample point we will approximate the spread by just getting the distance between
        the last main sample point and the the vartex of the trangle that we used for it and using it as a diameter of the sphere which volume we will use for weighting those points
    
    
    Implementation details:
    probably the bst is to get all of the sample point per tetrahedron to be calculated in sequence over single thread
    and then parallelize info on tetrahedrons 
    In case of using float32 and reserving space for shadow memory for enzyme we probably can keep 2-3 floats32 in shared memory per thread
    for x,y,z indicies of the supervoxels probably uint8 will be sufficient - morover we can unroll the loop
    we will also try to use generic names of the variables to keep track of the register memory usage 
    
    sv_center- indicates array length 3 with coordinates of point that are the center of the supervoxel
    tetr_dat - array of 5 points that first is sv center next 3 are verticies of the triangle that creates the base of the tetrahedron plus this triangle center so 5x3 array
    num_base_samp_points - how many main sample points we want to have between each triangle center and sv center in each tetrahedron
    shared_arr - array of length 3 where we have allocated space for temporarly storing the sample point that we are calculating
    out_sampled_points - array for storing the value of the sampled point that we got from interpolation and its weight that depends on the proximity to the edges of the tetrahedron and to other points
        hence the shape of the array is (num_base_samp_points+3)x5 where in second dimension first is the interpolated value of the point and second is its weight the last 3 entries are x,y,z coordinates of sampled point
    source_arr - array with image on which we are working    
"""
function get_sample_point_num(tetr_dat,num_base_samp_points,shared_arr,out_sampled_points,source_arr)
    samp_points=[]
    #TODO unroll the loop
    var1=0.0
    var2=0.0#will be used for example for interpolation
    #we iterate over rest of points in main sample points
    for point_num in UInt8(1):UInt8(num_base_samp_points)

        #we get the diffrence between the sv center and the triangle center
        shared_arr[1]= @get_diff_on_line_sv_tetr(1,5,point_num)
        shared_arr[2]=@get_diff_on_line_sv_tetr(2,5,point_num)
        shared_arr[3]=@get_diff_on_line_sv_tetr(3,5,point_num)

        ##calculate weight of the point
        #first distance from next and previous point on the line between sv center and triangle center
        var1=sqrt((shared_arr[1]/point_num)^2 +(shared_arr[2]/point_num)^2+(shared_arr[3]/point_num)^2)*2 #distance between main sample points (two times for distance to previous and next)
        #now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
        # we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
        for triangle_corner_num in UInt8(1):UInt8(3)
            #distance to the line between sv center and the  corner
            var1+=sqrt((shared_arr[1] -@get_diff_on_line_sv_tetr(1,triangle_corner_num+1,point_num) )^2
                           +(shared_arr[2] -@get_diff_on_line_sv_tetr(2,triangle_corner_num+1,point_num) )^2     
                           +(shared_arr[3] -@get_diff_on_line_sv_tetr(3,triangle_corner_num+1,point_num) )^2     
            ) 
        end#for triangle_corner_num     
        #now as we had looked into distance to other points in 5 directions we divide by 5 and save it to the out_sampled_points
        out_sampled_points[point_num,2]=var1/5

      
        ##time to get value by interpolation and save it to the out_sampled_points
        #now we get the location of sample point
        shared_arr[1]= tetr_dat[1,1]+shared_arr[1]
        shared_arr[2]= tetr_dat[1,2]+shared_arr[2]
        shared_arr[3]= tetr_dat[1,3]+shared_arr[3]
        #performing interpolation result is in var2 and it get data from shared_arr
        @threeDLinInterpol(source_arr)
        #saving the result of interpolated value to the out_sampled_points
        out_sampled_points[point_num,2]=var2
        #saving sample points coordinates
        out_sampled_points[point_num,3]=shared_arr[1]
        out_sampled_points[point_num,4]=shared_arr[2]
        out_sampled_points[point_num,5]=shared_arr[3]
        #for CPU debug
        push!(samp_points,copy(shared_arr))

    end#for num_base_samp_points

    ##### now we need to calculate the additional sample points that are branching out from the last main sample point
    for triangle_corner_num in UInt8(1):UInt8(3)
        #now we need to get diffrence between the last main sample point and the triangle corner
        shared_arr[1]=(tetr_dat[triangle_corner_num+1,1]-out_sampled_points[num_base_samp_points,3])*0.5
        shared_arr[2]=(tetr_dat[triangle_corner_num+1,2]-out_sampled_points[num_base_samp_points,4])*0.5
        shared_arr[3]=(tetr_dat[triangle_corner_num+1,3]-out_sampled_points[num_base_samp_points,5])*0.5


        out_sampled_points[num_base_samp_points+triangle_corner_num,2]=sqrt( shared_arr[1]^2+shared_arr[2]^2+shared_arr[3]^2)
        ##time to get value by interpolation and save it to the out_sampled_points
        #now we get the location of sample point
        shared_arr[1]= out_sampled_points[num_base_samp_points,3]+shared_arr[1]
        shared_arr[2]= out_sampled_points[num_base_samp_points,4]+shared_arr[2]
        shared_arr[3]= out_sampled_points[num_base_samp_points,5]+shared_arr[3]

        #performing interpolation result is in var2 and it get data from shared_arr
        @threeDLinInterpol(source_arr)
        #saving the result of interpolated value to the out_sampled_points
        out_sampled_points[num_base_samp_points+triangle_corner_num,2]=var2
        #saving sample points coordinates
        out_sampled_points[num_base_samp_points+triangle_corner_num,3]=shared_arr[1]
        out_sampled_points[num_base_samp_points+triangle_corner_num,4]=shared_arr[2]
        out_sampled_points[num_base_samp_points+triangle_corner_num,5]=shared_arr[3]


        #for CPU debug
        push!(samp_points,copy(shared_arr))

    end #for triangle_corner_num
    return samp_points
end




dims=(7,7,7)
dims_plus=(dims[1]+1,dims[2]+1,dims[3]+1)
radius=3.0
diam=radius*2
num_weights_per_point=6
example_set_of_svs=initialize_centers_and_control_points(dims,radius)
sv_centers,control_points=example_set_of_svs   # ,lin_x_add,lin_y_add,lin_z_add

# control_points first dimension is lin_x, lin_y, lin_z, oblique
# weights=zeros((dims_plus[1],dims_plus[2],dims_plus[3],num_weights_per_point))
weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
weights=weights.-0.5
weights=(weights).*100
weights = tanh.(weights*0.02)


# threads=(2,2,2)
# blocks=(2,2,2)

# control_points=call_apply_weights_to_locs_kern(CuArray(control_points),CuArray(weights),radius,threads,blocks)

# control_points=Array(control_points)

base_ind=(1,1,1)
tetrs= [get_tetrahedrons_of_sv(base_ind)
        # ,get_tetrahedrons_of_sv((2,1,1))
        # ,get_tetrahedrons_of_sv((1,2,1))
        # ,get_tetrahedrons_of_sv((1,1,2))
        
        # ,get_tetrahedrons_of_sv((1,2,2))
        # ,get_tetrahedrons_of_sv((2,2,1))
        # ,get_tetrahedrons_of_sv((1,2,1))
        # ,get_tetrahedrons_of_sv((2,2,2))

            ]

sv_center=sv_centers[base_ind[1],base_ind[2],base_ind[3],:]            
tetrs = collect(Iterators.flatten(tetrs))
tetrs = collect(Iterators.flatten(tetrs))
#augment triangles with their centers so last entry in a tuple is a triangle center and first is sv center
tetrs=map(tr->(tr[1],tr[2],tr[3],tr[4],get_triangle_center(tr)),tetrs)

#how many main sample points we want to have between each triangle center and sv center in each tetrahedron
num_base_samp_points=3


tetr_dat=collect(tetrs[1])
tetr_dat=invert(tetr_dat)
tetr_dat=combinedims(tetr_dat)

shared_arr=[0.0,0.0,0.0]
out_sampled_points=zeros((num_base_samp_points+3,5))

source_arr=zeros(Int.(dims.*(radius*2)))
points=get_sample_point_num(tetr_dat,num_base_samp_points,shared_arr,out_sampled_points,source_arr)

points_mesh_a=Meshes.Point3.(points)
tt=invert(splitdims(tetr_dat))
# tt=tt[2:4]
points_mesh_b=Meshes.Point3.(tt)
points_mesh=[points_mesh_a;points_mesh_b]


viz(points_mesh, color = 1:length(points_mesh))
# viz(points_mesh, color = [5,1,2,3,4])

"""
now we want to visualize the points that were selected for sampling and their weights
    we will display their weights by the spheres of the radius equal to weight
"""

# viz(tetrs, color = 1:length(tetrs))


"""
we want to check weather weighted sampling is working correctly by gettin base arr as the gaussian with known mean and variance
and checking weather our samples have similar weighted mean and variance
"""


function get_sample_point_num(tetr_dat,num_base_samp_points,shared_arr,out_sampled_points,source_arr)
    samp_points=[]
    #TODO unroll the loop
    var1=0.0
    var2=0.0#will be used for example for interpolation
    #we iterate over rest of points in main sample points
    for point_num in UInt8(1):UInt8(num_base_samp_points)

        #we get the diffrence between the sv center and the triangle center
        shared_arr[1]= @get_diff_on_line_sv_tetr(1,5,point_num)
        shared_arr[2]=@get_diff_on_line_sv_tetr(2,5,point_num)
        shared_arr[3]=@get_diff_on_line_sv_tetr(3,5,point_num)

        ##calculate weight of the point
        #first distance from next and previous point on the line between sv center and triangle center
        var1=sqrt((shared_arr[1]/point_num)^2 +(shared_arr[2]/point_num)^2+(shared_arr[3]/point_num)^2)*2 #distance between main sample points (two times for distance to previous and next)
        #now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
        # we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
        for triangle_corner_num in UInt8(1):UInt8(3)
            #distance to the line between sv center and the  corner
            var1+=sqrt((shared_arr[1] -@get_diff_on_line_sv_tetr(1,triangle_corner_num+1,point_num) )^2
                           +(shared_arr[2] -@get_diff_on_line_sv_tetr(2,triangle_corner_num+1,point_num) )^2     
                           +(shared_arr[3] -@get_diff_on_line_sv_tetr(3,triangle_corner_num+1,point_num) )^2     
            ) 
        end#for triangle_corner_num     
        #now as we had looked into distance to other points in 5 directions we divide by 5 and save it to the out_sampled_points
        out_sampled_points[point_num,2]=var1/5

      
        ##time to get value by interpolation and save it to the out_sampled_points
        #now we get the location of sample point
        shared_arr[1]= tetr_dat[1,1]+shared_arr[1]
        shared_arr[2]= tetr_dat[1,2]+shared_arr[2]
        shared_arr[3]= tetr_dat[1,3]+shared_arr[3]
        #performing interpolation result is in var2 and it get data from shared_arr
        @threeDLinInterpol(source_arr)
        #saving the result of interpolated value to the out_sampled_points
        out_sampled_points[point_num,2]=var2
        #saving sample points coordinates
        out_sampled_points[point_num,3]=shared_arr[1]
        out_sampled_points[point_num,4]=shared_arr[2]
        out_sampled_points[point_num,5]=shared_arr[3]
        #for CPU debug
        # push!(samp_points,copy(shared_arr))

    end#for num_base_samp_points

    last_samp_point=copy(shared_arr)
    print("last_samp_point $last_samp_point \n")
    ##### now we need to calculate the additional sample points that are branching out from the last main sample point
    for triangle_corner_num in UInt8(1):UInt8(3)
        #now we need to get diffrence between the last main sample point and the triangle corner
        shared_arr[1]=(tetr_dat[triangle_corner_num+1,1]-last_samp_point[1])*0.5
        shared_arr[2]=(tetr_dat[triangle_corner_num+1,2]-last_samp_point[2])*0.5
        shared_arr[3]=(tetr_dat[triangle_corner_num+1,3]-last_samp_point[3])*0.5

        # shared_arr[1]=@get_diff_on_line_last_tetr(1,triangle_corner_num+1)
        # shared_arr[2]=@get_diff_on_line_last_tetr(2,triangle_corner_num+1)
        # shared_arr[3]=@get_diff_on_line_last_tetr(3,triangle_corner_num+1)
        out_sampled_points[num_base_samp_points+triangle_corner_num,2]=sqrt( shared_arr[1]^2+shared_arr[2]^2+shared_arr[3]^2)
        ##time to get value by interpolation and save it to the out_sampled_points
        #now we get the location of sample point
        shared_arr[1]= last_samp_point[1]+shared_arr[1]
        shared_arr[2]= last_samp_point[2]+shared_arr[2]
        shared_arr[3]= last_samp_point[3]+shared_arr[3]

        #performing interpolation result is in var2 and it get data from shared_arr
        @threeDLinInterpol(source_arr)
        #saving the result of interpolated value to the out_sampled_points
        out_sampled_points[num_base_samp_points+triangle_corner_num,2]=var2
        #saving sample points coordinates
        out_sampled_points[num_base_samp_points+triangle_corner_num,3]=shared_arr[1]
        out_sampled_points[num_base_samp_points+triangle_corner_num,4]=shared_arr[2]
        out_sampled_points[num_base_samp_points+triangle_corner_num,5]=shared_arr[3]


        #for CPU debug
        push!(samp_points,copy(shared_arr))

    end #for triangle_corner_num
    return samp_points
end