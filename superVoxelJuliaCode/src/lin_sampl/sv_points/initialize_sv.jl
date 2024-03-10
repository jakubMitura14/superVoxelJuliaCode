
using SplitApplyCombine

"""
get 4 dimensional array of cartesian indicies of a 3 dimensional array
thats size is passed as an argument dims
"""
function get_base_indicies_arr(dims)    
    indices = CartesianIndices(dims)
    # indices=collect.(Tuple.(collect(indices)))
    indices=Tuple.(collect(indices))
    indices=collect(Iterators.flatten(indices))
    indices=reshape(indices,(3,dims[1],dims[2],dims[3]))
    indices=permutedims(indices,(2,3,4,1))
    return indices
end#get_base_indicies_arr


function get_corrected_dim(ax,radius,image_shape)
    diam=radius*2
    return Int(ceil((image_shape[ax]-5)/diam))-2
end    

function get_dif(ax,image_shape,dims,diam)
    return max(floor((image_shape[ax]-((dims[ax]+1).*diam))/2),2.0)
end

"""
initialize sv centers coordinates 
"""
function get_sv_centers(radius,image_shape)
    diam=radius*2
    dims=(get_corrected_dim(1,radius,image_shape),get_corrected_dim(2,radius,image_shape),get_corrected_dim(3,radius,image_shape))
    diffs= (get_dif(1,image_shape,dims,diam),get_dif(2,image_shape,dims,diam),get_dif(3,image_shape,dims,diam))
    # diffs= (1.0,1.0,1.0)
    res= get_base_indicies_arr(dims)*diam
    res[:,:,:,1]=res[:,:,:,1].+diffs[1]
    res[:,:,:,2]=res[:,:,:,2].+diffs[2]
    res[:,:,:,3]=res[:,:,:,3].+diffs[3]
    return res,dims,diffs

end



"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
we get linear control points - so points that are on the lines between each of the sv_centers - hence their modifications will require just one weight
we will create linear points by moving by radius in each axis
"""
function get_linear_control_points(dims,axis,diam,radius,diffs)
    #increasing dimension as we need to have them both up and down the axis
    # dim_new=collect(Iterators.flatten(dims))#.+1
    # dim_new[axis]=dim_new[axis]+1
    dim_new=collect(Iterators.flatten(dims)).+1
    indicies=get_base_indicies_arr(Tuple(dim_new)).-1
    indicies=indicies.*diam
    indicies=indicies.+diam
    indicies_ax=indicies[:,:,:,axis].-radius
    indicies[:,:,:,axis]=indicies_ax
    res=indicies
    res[:,:,:,1]=res[:,:,:,1].+diffs[1]
    res[:,:,:,2]=res[:,:,:,2].+diffs[2]
    res[:,:,:,3]=res[:,:,:,3].+diffs[3]

    return res
end#get_linear_control_points


"""
will get oblique control points - so points that are on the corners of the cube that is enclosing a volume of
non modified supervoxel area 
"""
function get_oblique_control_points(dims,diam,radius,diffs)
    #increasing dimension as we need to have them both up and down the axis
    dim_new=collect(Iterators.flatten(dims)).+1
    indicies=get_base_indicies_arr(Tuple(dim_new)).-1
    indicies=indicies.*diam
    res= indicies.+radius
    res[:,:,:,1]=res[:,:,:,1].+diffs[1]
    res[:,:,:,2]=res[:,:,:,2].+diffs[2]
    res[:,:,:,3]=res[:,:,:,3].+diffs[3]

    return res


end#get_oblique_control_points


"""
flips the value of the index of the tuple at the position ind needed for get_linear_between function
"""
function flip_num(base_ind,tupl,ind)
    arr=collect(tupl)
    # arr=append!(arr,[4])
    if(arr[ind]==base_ind[ind])
        arr[ind]=base_ind[ind]+1
    else
        arr[ind]=base_ind[ind]
    end    
    return arr
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
        return [ind_1[1],base_ind[2],base_ind[3],1]
    end
    if(ind_1[2]==ind_2[2])
        return [base_ind[1],ind_1[2],base_ind[3],2]
    end

    return [base_ind[1],base_ind[2],ind_1[3],3]
end



"""
from control points and base index of the supervoxel we get the tetrahedron that is created by the center of the supervoxel
and the control points we are creating here all tetrahedrons that cover one of the corners of the cube
"""
function get_tetr_triangles_in_corner(base_ind,corner)
    corner=Float32.(append!(collect(corner),[4]))
    
    sv_center=Float32.([base_ind[1],base_ind[2],base_ind[3],-1.0])
    p_a=Float32.(flip_num(base_ind,corner,1))
    p_b=Float32.(flip_num(base_ind,corner,2))
    p_c=Float32.(flip_num(base_ind,corner,3))


    p_ab=Float32.(get_linear_between(base_ind,p_a,p_b))
    p_ac=Float32.(get_linear_between(base_ind,p_a,p_c))
    p_bc=Float32.(get_linear_between(base_ind,p_b,p_c))

    dummy=Float32.([-1.0,-1.0,-1.0,-1.0])
    
    res= [[sv_center;;corner;;p_a;;p_ab;;dummy]
        ,[sv_center;;corner;;p_ab;;p_b;;dummy]
        ,[sv_center;;corner;;p_b;;p_bc;;dummy]
        ,[sv_center;;corner;;p_bc;;p_c;;dummy]
        ,[sv_center;;corner;;p_a;;p_ac;;dummy]
        ,[sv_center;;corner;;p_ac;;p_c;;dummy] 
    ]

    # res=permutedims(res, (2, 1))
    # print("uuuuuuuuu $(size(res))")
    # res=reshape(io,(5,3,6))    
    # res=permutedims(res, (3,1,2))
    res=map(el-> permutedims(el, (2, 1)) ,res)
    res=map(el-> reshape(el, (1, size(el)...)) ,res)
    res=vcat(res...)
    return res

end



"""
given indicies of current supervoxel and control points we get all triangles that are covering the surface of the supervoxel
"""
function get_all_surface_triangles_of_sv(base_ind)
    return [
        get_tetr_triangles_in_corner(base_ind,(base_ind[1],base_ind[2],base_ind[3]))
        ,get_tetr_triangles_in_corner(base_ind,(base_ind[1]+1,base_ind[2]+1,base_ind[3]))
        ,get_tetr_triangles_in_corner(base_ind,(base_ind[1],base_ind[2]+1,base_ind[3]+1))
        ,get_tetr_triangles_in_corner(base_ind,(base_ind[1]+1,base_ind[2],base_ind[3]+1))
    ]
end #get_tetrahedrons_of_sv

"""
get a flattened array of all surface triangles of all supervoxels
in first dimension every 24 elements are a single supervoxel
second dimension is size 5 and is in orde sv_center, point a,point b,point c,centroid 
    where centroid is a placeholder for centroid of the triangle a,b,c
in last dimension we have x,y,z coordinates of the point
currently we have just indicies to the appropriate arrays -> it need to be populated after weights get applied        
"""
function get_flattened_triangle_data(dims)
    indices = CartesianIndices(dims)
    # indices=collect.(Tuple.(collect(indices)))
    indices=Tuple.(collect(indices))
    indices=collect(Iterators.flatten(indices))
    indices=reshape(indices,(3,dims[1]*dims[2]*dims[3]))
    indices=permutedims(indices,(2,1))

    indices=splitdims(indices,1)

    all_surf_triangles=map(el->get_all_surface_triangles_of_sv(el),indices)
    #concatenate all on first dimension
    all_surf_triangles=map(el->vcat(el...),all_surf_triangles)
    all_surf_triangles=vcat(all_surf_triangles...)


    return all_surf_triangles
end


"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
and the radius of supervoxels will return the grid of points that will be used as centers of supervoxels 
and the intilia positions of the control points
"""
function initialize_centers_and_control_points(image_shape,radius)
    diam=radius*2
    sv_centers,dims,diffs= get_sv_centers(radius,image_shape)
    lin_x=get_linear_control_points(dims,1,diam,radius,diffs)
    lin_y=get_linear_control_points(dims,2,diam,radius,diffs)
    lin_z=get_linear_control_points(dims,3,diam,radius,diffs)
    oblique=get_oblique_control_points(dims,diam,radius,diffs)

    flattened_triangles=get_flattened_triangle_data(dims)  

    return sv_centers,combinedims([lin_x, lin_y, lin_z, oblique],4),flattened_triangles,dims
end#initialize_centeris_and_control_points    
    


# radius=4.0
# diam=radius*2
# # a=36
# a=42
# # image_shape=(36,37,38)
# image_shape=(a,a,a)
# dims=(get_corrected_dim(1,radius,image_shape),get_corrected_dim(2,radius,image_shape),get_corrected_dim(3,radius,image_shape))
# diffs= floor.((image_shape.-((dims.+1).*diam))./2).+1
# # diffs= (1,1,1)

# sv_centers=(get_base_indicies_arr(dims)*diam)


# sv_centers[:,:,:,1]=sv_centers[:,:,:,1].+diffs[1]
# sv_centers[:,:,:,2]=sv_centers[:,:,:,2].+diffs[2]
# sv_centers[:,:,:,3]=sv_centers[:,:,:,3].+diffs[3]

# maximum(sv_centers)
# minimum(sv_centers)