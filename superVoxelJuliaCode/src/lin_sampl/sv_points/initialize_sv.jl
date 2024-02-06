"""
initializing supervoxel points
"""


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


"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
we get linear control points - so points that are on the lines between each of the sv_centers - hence their modifications will require just one weight
we will create linear points by moving by radius in each axis
"""
function get_linear_control_points(dims,axis,diam,radius)
    #increasing dimension as we need to have them both up and down the axis
    dim_new=collect(Iterators.flatten(dims))#.+1
    dim_new[axis]=dim_new[axis]+1
    indicies=get_base_indicies_arr(Tuple(dim_new)).-1
    indicies=indicies.*diam
    indicies_ax=indicies[:,:,:,axis].+radius
    indicies[:,:,:,axis]=indicies_ax
    return indicies
end#get_linear_control_points


"""
will get oblique control points - so points that are on the corners of the cube that is enclosing a volume of
non modified supervoxel area 
"""
function get_oblique_control_points(dims,diam,radius)
    #increasing dimension as we need to have them both up and down the axis
    dim_new=collect(Iterators.flatten(dims)).+1
    indicies=get_base_indicies_arr(Tuple(dim_new)).-1
    indicies=indicies.*diam
    return indicies.+radius
end#get_oblique_control_points

function get_linear_control_points_added(dims,axis,diam,radius)
    #increasing dimension as we need to have them both up and down the axis
    indicies=get_oblique_control_points(dims,diam,radius)
    indicies_ax=indicies[:,:,:,axis].-radius
    indicies[:,:,:,axis]=indicies_ax
    return indicies
end#get_linear_control_points




"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
and the radius of supervoxels will return the grid of points that will be used as centers of supervoxels 
and the intilia positions of the control points
"""
function initialize_centers_and_control_points(dims,radius)
    diam=radius*2

    sv_centers=get_base_indicies_arr(dims)*diam
    lin_x=get_linear_control_points(dims,1,diam,radius)
    lin_y=get_linear_control_points(dims,2,diam,radius)
    lin_z=get_linear_control_points(dims,3,diam,radius)
    oblique=get_oblique_control_points(dims,diam,radius)

    lin_x_add=get_linear_control_points_added(dims,1,diam,radius)
    lin_y_add=get_linear_control_points_added(dims,2,diam,radius)
    lin_z_add=get_linear_control_points_added(dims,3,diam,radius)

    return sv_centers,lin_x,lin_y,lin_z,oblique,lin_x_add,lin_y_add,lin_z_add


end#initialize_centers_and_control_points    

# dims=(4,4,4)
# collect(Iterators.flatten(dims)).+1


# dims=(2,2,2)
# axis=3
# diam=4.0
# radius=2.0
# uu=get_linear_control_points(dims,axis,diam,radius)

# uu[1,1,1,:]