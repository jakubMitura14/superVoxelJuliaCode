# """
# we have basic grid of points that will constitute the centers of supervoxels - this grid is constant
# basic control points will be also on lines between each of the sv_centers so we will have 3*sv_centers amount of control points
# plus the additional layer in each axis 

# next is a control point in oblique direction that is common for 6 neighbouting sv_centers where those sv centers create a cube
#     so in order to get any point in a cube we need to move on the 3 of the edges of the cube what will give us x y and z coordinates
#     for oblique control points we will just need to get a line between sv center bolow and above for x coordinate and so on for y and z

# important is to always be able to draw a line between supervoxel center and its control point without leaving supervoxel volume - as the later sampling will depend on it
#     we want to keep this star shaped polyhedra ... 

# """
using Pkg
using ChainRulesCore,Zygote,CUDA,Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux,LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays
using LinearAlgebra
using Revise
using Images,ImageFiltering


function  get_point_on_a_line(vertex_0,vertex_1,weight)
    diff_x=vertex_1[1]-vertex_0[1]
    diff_y=vertex_1[2]-vertex_0[2]
    diff_z=vertex_1[3]-vertex_0[3]
    return [vertex_0[1]+(diff_x*weight),vertex_0[2]+(diff_y*weight),vertex_0[3]+(diff_z*weight)]
end

"""
we want to use a list of weights to move along a line between alowable maximum and minimum of points positions
for linear it is simply moving over a line between adjacent sv centers along each axis
for oblique it is selecting a position in a cube that is created by 8 sv centers

the function is designed to be broadcasted over the list of weights and the list of initialized 
control points - so we will get a list of new points that will be used as final control points to get borders of supervoxels
generally control points that are input are initialized to be in the center position of their allowable positions between minimum and maximum
morover basic control points that are modified by this function are independent one from another; and can move freely

    important we assume we have list of weights in a range from -1 to 1 (so for example after tanh) 
    also we assume that we have just a vector of weights and a set of single points - as we will broadcast over point arrays
     control_points first dimension is lin_x, lin_y, lin_z, oblique
"""
function apply_weights_to_locs(control_points,weights,radius)
    return [ [control_points[1,1]+weights[1]*radius,control_points[1,2],control_points[1,3]]#lin_x
            ,[control_points[2,1],control_points[2,2]+weights[2]*radius,control_points[2,3]]#lin_y
            ,[control_points[3,1],control_points[3,2],control_points[3,3]+weights[3]*radius]#lin_z
            ,[control_points[4,1]+weights[4]*radius,control_points[4,2]+weights[5]*radius,control_points[4,3]+weights[6]*radius]#oblique
    ]
end #apply_weights_to_locs

function apply_weights_to_locs_kern(control_points,weights,radius)
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) 

    control_points[x,y,z,1,1]=control_points[x,y,z,1,1]+weights[x,y,z,1]*radius#lin_x
    control_points[x,y,z,2,2]=control_points[x,y,z,2,2]+weights[x,y,z,2]*radius#lin_y
    control_points[x,y,z,3,3]=control_points[x,y,z,3,3]+weights[x,y,z,3]*radius#lin_z
    
    control_points[x,y,z,4,1]=control_points[x,y,z,4,1]+weights[x,y,z,4]*radius
    control_points[x,y,z,4,2]=control_points[x,y,z,4,2]+weights[x,y,z,5]*radius
    control_points[x,y,z,4,3]=control_points[x,y,z,4,3]+weights[x,y,z,6]*radius
    return nothing

end #apply_weights_to_locs

############################# Enzyme differentiation


function apply_weights_to_locs_kern_deff(control_points,d_control_points,weights,d_weights,radius)
    Enzyme.autodiff_deferred(Reverse,apply_weights_to_locs_kern, Const, Duplicated(control_points, d_control_points),Duplicated(weights, d_weights),Const(radius) )
    return nothing
end


function call_apply_weights_to_locs_kern(control_points,weights,radius,threads,blocks)

    @cuda threads = threads blocks = blocks apply_weights_to_locs_kern(control_points,weights,radius)
    return control_points
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_apply_weights_to_locs_kern),control_points,weights,radius,threads,blocks)
    

    control_points_out = call_apply_weights_to_locs_kern(control_points,weights,radius,threads,blocks)

    function kernel1_pullback(d_control_points_out)

        d_weights = CUDA.ones(size(d_weights))

        @cuda threads = threads blocks = blocks apply_weights_to_locs_kern_deff(control_points,CuArray(collect(d_control_points_out)),weights,d_weights,radius)

        return d_control_points,d_weights,NoTangent(),NoTangent(),NoTangent()
    end   
    return control_points_out, kernel1_pullback

end


############## lux definitions
struct Points_weights_str<: Lux.AbstractExplicitLayer
    radius::Int
    threads::Tuple{Int,Int,Int}
    blocks::Tuple{Int,Int,Int}
end

function Points_weights(radius,threads,blocks)
    return Points_weights_str(radius,threads,blocks)
end

function Lux.initialparameters(rng::AbstractRNG, l::Points_weights_str)
    return ()
end

function Lux.initialstates(::AbstractRNG, l::Points_weights_str)::NamedTuple
    return (radius =l.radius,threads=l.threads,blocks=l.blocks )
end

function (l::Points_weights_str)(x, ps, st::NamedTuple)
    control_points,weights= x
    return call_apply_weights_to_locs_kern(control_points,weights,st.radius,st.threads,st.blocks),st
end







# function dummy_fun(pa,pb)
#     print("pa $(pa) pb $(pb) \n")
#     return pa[1]+1,pb[1]-1
# end

# arra=[[1.0],[1.0],[1.0]]
# arrb=[[1.0],[1.0],[1.0]]

# broadcast((a, b) -> dummy_fun(a, b), arra, arrb)


# aa

# get_point_on_a_line_b((1.0,2.0,1.0),(2.0,2.0,2.0),0.5)


# def get_point_inside_triange(vertex_a,vertex_b,vertex_c,edge_weights):
#     """ 
#     we want to put a new point in a triangle - that will be a new control point
#     point is as specified constrained by a triangle weights live on two of the primary triangle edges
#     so we take  2 edges establish positions of temporary points by moving on those edges by percentege of their length
#     then we get a line between those new points and apply 3rd weight to it so we will move along this new line
#     """
#     p0=get_point_on_a_line_b(vertex_a,vertex_b,edge_weights[0])
#     p1=get_point_on_a_line_b(vertex_b,vertex_c,edge_weights[1])

#     res=get_point_on_a_line_b(p0,p1,edge_weights[2])
#     return jnp.array(res)


# def get_point_inside_square(vertex_a,vertex_b,vertex_c,vertex_d,edge_weights):
#     """ 
#     we want to put a new point in a square - that will be a new control point
#     we will need just to get a point on each edge - connect points from opposing edges by the line 
#     and find an intersection point of those lines
#     """
#     p0=get_point_on_a_line_b(vertex_a,vertex_b,edge_weights[0])
#     p1=get_point_on_a_line_b(vertex_c,vertex_d,edge_weights[1])
    
#     p2=get_point_on_a_line_b(vertex_a,vertex_d,edge_weights[2])
#     p3=get_point_on_a_line_b(vertex_b,vertex_c,edge_weights[3])

#     return lineLineIntersection(p0,p1,p2,p3)


#     def lineLineIntersection(A, B, C, D):
#     """ 
#     based on https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
#     """
#     # Line AB represented as a1x + b1y = c1
#     a1 = B[1] - A[1]
#     b1 = A[0] - B[0]
#     c1 = a1*(A[0]) + b1*(A[1])
 
#     # Line CD represented as a2x + b2y = c2
#     a2 = D[1] - C[1]
#     b2 = C[0] - D[0]
#     c2 = a2*(C[0]) + b2*(C[1])
    

#     determinant = (a1*b2 - a2*b1)+0.000000000001
    
#     # if (determinant == 0):
#     #     # The lines are parallel. This is simplified
#     #     # by returning a pair of FLT_MAX
#     #     return Point(10**9, 10**9)
#     # else:
#     x = (b2*c1 - b1*c2)/determinant
#     y = (a1*c2 - a2*c1)/determinant
#     return [x, y]



