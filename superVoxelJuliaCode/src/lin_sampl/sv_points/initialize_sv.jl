
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


"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
we get linear control points - so points that are on the lines between each of the sv_centers - hence their modifications will require just one weight
we will create linear points by moving by radius in each axis
"""
function get_linear_control_points(dims,axis,diam,radius)
    #increasing dimension as we need to have them both up and down the axis
    # dim_new=collect(Iterators.flatten(dims))#.+1
    # dim_new[axis]=dim_new[axis]+1
    dim_new=collect(Iterators.flatten(dims)).+1
    indicies=get_base_indicies_arr(Tuple(dim_new)).-1
    indicies=indicies.*diam
    indicies=indicies.+diam
    indicies_ax=indicies[:,:,:,axis].-radius
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
flips the value of the index of the tuple at the position ind needed for get_linear_between function
"""
function flip_num(base_ind,tupl,ind)
    arr=collect(tupl)
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
        return [ind_1[1],base_ind[2],base_ind[3]]
    end
    if(ind_1[2]==ind_2[2])
        return [base_ind[1],ind_1[2],base_ind[3]]
    end

    return [base_ind[1],base_ind[2],ind_1[3]]
end



"""
from control points and base index of the supervoxel we get the tetrahedron that is created by the center of the supervoxel
and the control points we are creating here all tetrahedrons that cover one of the corners of the cube
"""
function get_tetr_triangles_in_corner(base_ind,corner)
    corner=Float32.(collect(corner))
    sv_center=Float32.([base_ind[1],base_ind[2],base_ind[3]])
    p_a=Float32.(flip_num(base_ind,corner,1))
    p_b=Float32.(flip_num(base_ind,corner,2))
    p_c=Float32.(flip_num(base_ind,corner,3))


    p_ab=Float32.(get_linear_between(base_ind,p_a,p_b))
    p_ac=Float32.(get_linear_between(base_ind,p_a,p_c))
    p_bc=Float32.(get_linear_between(base_ind,p_b,p_c))

    dummy=Float32.([-1.0,-1.0,-1.0])
    res= [sv_center;;corner;;p_a;;p_ab;;dummy;;
        sv_center;;corner;;p_ab;;p_b;;dummy;;
        sv_center;;corner;;p_b;;p_bc;;dummy;; 
        sv_center;;corner;;p_bc;;p_c;;dummy;;
        sv_center;;corner;;p_a;;p_ac;;dummy;;
        sv_center;;corner;;p_ac;;p_c;;dummy;; 
    ]
    # res=permutedims(res, (2, 1))
    # print("uuuuuuuuu $(size(res))")
    # res=reshape(io,(5,3,6))
    
    # res=permutedims(res, (3,1,2))

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
    #get the output of single get_tetr_triangles_in_corner to order
    all_surf_triangles=map(el_out->map(el_in->permutedims(el_in, (2, 1)),el_out) ,all_surf_triangles)
    all_surf_triangles=map(el_out->map(el_in->reshape(el_in,(5,3,6)),el_out) ,all_surf_triangles)
    all_surf_triangles=map(el_out->map(el_in->permutedims(el_in, (3,1,2)),el_out) ,all_surf_triangles)
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
function initialize_centers_and_control_points(dims,radius)
    diam=radius*2

    sv_centers=get_base_indicies_arr(dims)*diam
    lin_x=get_linear_control_points(dims,1,diam,radius)
    lin_y=get_linear_control_points(dims,2,diam,radius)
    lin_z=get_linear_control_points(dims,3,diam,radius)
    oblique=get_oblique_control_points(dims,diam,radius)
    flattened_triangles=get_flattened_triangle_data(dims)
    
    return sv_centers,combinedims([lin_x, lin_y, lin_z, oblique],4),flattened_triangles
end#initialize_centeris_and_control_points    
    


# dims=(4,4,4)
# collect(Iterators.flatten(dims)).+1


# dims=(2,2,2)
# axis=3
# diam=4.0
# radius=2.0
# uu=get_linear_control_points(dims,axis,diam,radius)

# uu[1,1,1,:]