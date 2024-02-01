"""
initializing supervoxel points
"""


# function get_grid_c_points(cfg)
#     """ 
#     setting up initial sv centers (grid_a_points) and control points
#      grid_c_points - at the intersections of the sv corners
#      important - output is not batched
#     """
#     r=cfg.r
#     half_r=r/2
#     diam_x=cfg.img_size[1]+r
#     diam_y=cfg.img_size[2]+r
    
#     # gridd=einops.rearrange(jnp.mgrid[r:diam_x:r, r:diam_y:r],'c x y-> x y c')-half_r
#     gridd_bigger=einops.rearrange(jnp.mgrid[0:diam_x+r:r,0:diam_y+r:r],'c x y-> x y c')-half_r
#     grid_c_points=(gridd_bigger+jnp.array([half_r,half_r]))[0:-1,0:-1,:]

#     return grid_c_points

# using LinearAlgebra

function get_cartesian_indices(dims)
    return CartesianIndices(dims)
end

dims = (10, 10, 10)
indices = CartesianIndices(dims)
# indices=collect.(Tuple.(collect(indices)))
indices=Tuple.(collect(indices))
indices=collect(Iterators.flatten(indices))
indices=reshape(indices,(3,dims[1],dims[2],dims[3]))
indices=permutedims(indices,(2,3,4,1))
indices[9,3,4,:]

Array.(indices)

"""
we have basic grid of points that will constitute the centers of supervoxels - this grid is constant
basic control points will be also on lines between each of the sv_centers so we will have 3*sv_centers amount of control points
plus the additional layer in each axis 

next we will get free control points that will be using projections on the basis of basic control points and the sv_centers

next is a control point in oblique direction that is common for 6 neighbouting sv_centers where those sv centers create a cube
    so in order to get any point in a cube we need to move on the 3 of the edges of the cube what will give us x y and z coordinates

Then we need to get additional points between current control points - so oblique control points can be thought as 
apices of pyramids thats base are cotrnol points between sv centers (linear ones) - now we choose a point in a triangular face of the pyramid
then get a line between supervoxel ceneters of two supervoxels that share the wall and move the point in direction of one or the other we can get multiple such points

the spetial case are the lines of the edges of the pyramid - as in those lines volumes of 3 supervoxels meet - so if we want to modify it - it require spetial handling to
    avoid compromising star shape like volume ... 

important is to always be able to draw a line between supervoxel center and its control point without leaving supervoxel volume - as the later sampling will depend on it
    we want to keep this star shaped polyhedra ... 

"""



def lineLineIntersection(A, B, C, D):
    """ 
    based on https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
    """
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*(A[0]) + b1*(A[1])
 
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*(C[0]) + b2*(C[1])
    

    determinant = (a1*b2 - a2*b1)+0.000000000001
    
    # if (determinant == 0):
    #     # The lines are parallel. This is simplified
    #     # by returning a pair of FLT_MAX
    #     return Point(10**9, 10**9)
    # else:
    x = (b2*c1 - b1*c2)/determinant
    y = (a1*c2 - a2*c1)/determinant
    return [x, y]





function  get_point_on_a_line_b(vertex_0,vertex_1,weight)
    diff_x=vertex_1[1]-vertex_0[1]
    diff_y=vertex_1[2]-vertex_0[2]
    diff_z=vertex_1[3]-vertex_0[3]
    return [vertex_0[1]+(diff_x*weight),vertex_0[2]+(diff_y*weight),vertex_0[3]+(diff_z*weight)]
end

get_point_on_a_line_b((1.0,2.0,1.0),(2.0,2.0,2.0),0.5)


def get_point_inside_triange(vertex_a,vertex_b,vertex_c,edge_weights):
    """ 
    we want to put a new point in a triangle - that will be a new control point
    point is as specified constrained by a triangle weights live on two of the primary triangle edges
    so we take  2 edges establish positions of temporary points by moving on those edges by percentege of their length
    then we get a line between those new points and apply 3rd weight to it so we will move along this new line
    """
    p0=get_point_on_a_line_b(vertex_a,vertex_b,edge_weights[0])
    p1=get_point_on_a_line_b(vertex_b,vertex_c,edge_weights[1])

    res=get_point_on_a_line_b(p0,p1,edge_weights[2])
    return jnp.array(res)


def get_point_inside_square(vertex_a,vertex_b,vertex_c,vertex_d,edge_weights):
    """ 
    we want to put a new point in a square - that will be a new control point
    we will need just to get a point on each edge - connect points from opposing edges by the line 
    and find an intersection point of those lines
    """
    p0=get_point_on_a_line_b(vertex_a,vertex_b,edge_weights[0])
    p1=get_point_on_a_line_b(vertex_c,vertex_d,edge_weights[1])
    
    p2=get_point_on_a_line_b(vertex_a,vertex_d,edge_weights[2])
    p3=get_point_on_a_line_b(vertex_b,vertex_c,edge_weights[3])

    return lineLineIntersection(p0,p1,p2,p3)