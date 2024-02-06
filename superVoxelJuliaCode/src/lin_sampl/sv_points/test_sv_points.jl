"""
main idea is to check weather a line that goes from sv center to control point is inside the supervoxel
In order to test it we need to define a mesh which surface is defined by the set of control points

Plan is to define in meshes.jl the shape using a set of tetrahydras - we then need to check weather they intersect (they should not)
we can then later also test sampling scheme and check if a sampled point is inside any of the tetrahedron 
meshes additionally supply the visualization functionalities
"""

using Revise
using Meshes
using LinearAlgebra
using GLMakie
using Combinatorics

includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")

dims=(3,3,3)
radius=3.0

example_set_of_svs=initialize_centers_and_control_points(dims,radius)
sv_centers,lin_x,lin_y,lin_z,oblique,lin_x_add,lin_y_add,lin_z_add=example_set_of_svs


sv_center=Meshes.Point3(sv_centers[1,1,1,:])


lin_x_point_pre=Meshes.Point3(lin_x[1,1,1,:])
lin_x_point_post=Meshes.Point3(lin_x[2,1,1,:])

lin_y_point_pre=Meshes.Point3(lin_y[1,1,1,:])
lin_y_point_post=Meshes.Point3(lin_y[1,2,1,:])

lin_z_point_pre=Meshes.Point3(lin_z[1,1,1,:])
lin_z_point_post=Meshes.Point3(lin_z[1,1,2,:])

lin_x_point_pre_add=Meshes.Point3(lin_x_add[2,1,1,:])
# lin_x_point_pre_add=Meshes.Point3(lin_x_add[1,2,2,:])
lin_x_point_post_add=Meshes.Point3(lin_x_add[2,2,2,:])

lin_y_point_pre_add=Meshes.Point3(lin_y_add[1,2,1,:])
lin_y_point_post_add=Meshes.Point3(lin_y_add[2,2,2,:])

lin_z_point_pre_add=Meshes.Point3(lin_z_add[1,1,2,:])
lin_z_point_post_add=Meshes.Point3(lin_z_add[2,2,2,:])



oblique_point_1=Meshes.Point3(oblique[1,1,1,:])
oblique_point_2=Meshes.Point3(oblique[2,1,1,:])
oblique_point_3=Meshes.Point3(oblique[1,2,1,:])
oblique_point_4=Meshes.Point3(oblique[1,1,2,:])
oblique_point_5=Meshes.Point3(oblique[1,2,2,:])
oblique_point_6=Meshes.Point3(oblique[2,2,1,:])
oblique_point_7=Meshes.Point3(oblique[2,1,2,:])
oblique_point_8=Meshes.Point3(oblique[2,2,2,:])


#take a corner and can modify just one of the coordinates will get 3 combinations
1,1,1
2,1,1
1,2,1

1,1,1
2,1,1
1,1,2

1,1,1
1,1,2
1,2,1



1,2,1
1,2,2
2,2,2

1,2,1
2,2,1
2,2,2

1,2,2
2,1,2
2,2,2

2,2,1
2,1,2
2,2,2

#sv_center,
points=[lin_x_point_pre,lin_x_point_post
,lin_y_point_pre,lin_y_point_post
,lin_z_point_pre,lin_z_point_post
,oblique_point_1,oblique_point_2,oblique_point_3,oblique_point_4
,oblique_point_5,oblique_point_6,oblique_point_7,oblique_point_8]

# we will create a bunch of tetrahedrons on the basis of the points above
function is_point_in_array(x,y,z,arr)
    p=Meshes.Point(x,y,z)
    sizz=size(arr)
    ob= reshape(arr,sizz[1]*sizz[2]*sizz[3],3)
    ob=map(el-> Meshes.Point(el[1],el[2],el[3]),eachrow(ob))
    # map(el-> print(el),eachrow(ob))
    res=p in ob
    return res
end     

x=9.0
y=6.0
z=9.0
is_point_in_array(x,y,z,oblique)
is_point_in_array(x,y,z,lin_x)
is_point_in_array(x,y,z,lin_y)
is_point_in_array(x,y,z,lin_z)

is_point_in_array(x,y,z,lin_z_add)
is_point_in_array(x,y,z,lin_z_add)
is_point_in_array(x,y,z,lin_z_add)

function flip_num(tupl,ind)
    arr=collect(tupl)
    if(arr[ind]==1)
        arr[ind]=2
    else
        arr[ind]=1
    end    
    return Tuple(arr)
end

function get_tetr_a(corner)
    p_a=flip_num(corner,1)
    p_b=flip_num(corner,2)
    p_c=flip_num(corner,3)
    
    print("p_a $(p_a) p_b $(p_b) p_c $(p_c)")
    p_a=Meshes.Point3(oblique[p_a[1],p_a[2],p_a[3],:])
    p_b=Meshes.Point3(oblique[p_b[1],p_b[2],p_b[3],:])
    p_c=Meshes.Point3(oblique[p_c[1],p_c[2],p_c[3],:])
    corner=Meshes.Point3(oblique[corner[1],corner[2],corner[3],:])
    return [Meshes.Tetrahedron(sv_center ,corner,p_a,p_b)
    ,Meshes.Tetrahedron(sv_center ,corner,p_b,p_c)
    ,Meshes.Tetrahedron(sv_center ,corner,p_a,p_c)
    ]
end


tetrs=[
    get_tetr_a((1,1,1))
    ,get_tetr_a((2,2,1))
    ,get_tetr_a((1,2,2))
    ,get_tetr_a((2,1,2))

]
tetrs = collect(Iterators.flatten(tetrs))
viz(tetrs, color = 1:length(tetrs))


tt=get_tetr_a((1,1,1))
tetrs=[
    tt[1],tt[2],tt[3]
#  Meshes.Tetrahedron(sv_center,oblique_point_1,oblique_point_2,oblique_point_3)
#  Meshes.Tetrahedron(sv_center,oblique_point_1,oblique_point_2,oblique_point_4)

#  Meshes.Tetrahedron(sv_center,oblique_point_3,oblique_point_5,oblique_point_8)
#  Meshes.Tetrahedron(sv_center,oblique_point_6,oblique_point_3,oblique_point_8)

#  Meshes.Tetrahedron(sv_center,oblique_point_5,oblique_point_7,oblique_point_8)
#  Meshes.Tetrahedron(sv_center,oblique_point_6,oblique_point_7,oblique_point_8)

# Meshes.Tetrahedron(sv_center,Meshes.Point3(oblique[1,1,1,:]),Meshes.Point3(oblique[2,1,1,:]),Meshes.Point3(oblique[1,2,1,:]))
# Meshes.Tetrahedron(sv_center,Meshes.Point3(oblique[1,1,1,:]),Meshes.Point3(oblique[2,1,1,:]),Meshes.Point3(oblique[1,1,2,:]))
# Meshes.Tetrahedron(sv_center,Meshes.Point3(oblique[1,1,1,:]),Meshes.Point3(oblique[1,1,2,:]),Meshes.Point3(oblique[1,2,1,:]))

# Meshes.Tetrahedron(sv_center,Meshes.Point3(oblique[1,2,2,:]),Meshes.Point3(oblique[2,2,2,:]),Meshes.Point3(oblique[1,2,1,:]))
# Meshes.Tetrahedron(sv_center,Meshes.Point3(oblique[1,2,2,:]),Meshes.Point3(oblique[1,2,1,:]),Meshes.Point3(oblique[1,1,2,:]))
# Meshes.Tetrahedron(sv_center,Meshes.Point3(oblique[1,2,2,:]),Meshes.Point3(oblique[2,2,2,:]),Meshes.Point3(oblique[1,1,2,:]))




# 3,3,9    3,9,3
]

viz(tetrs, color = 1:length(tetrs))



sv_center ∈ tetrs[5]

"""
we need to check weather points in the cube defined by the oblique points  are all in some tetrahedron
also we need to check weather they are not in multiple tetrahedrons at once; 
Hovewer tetrahedron walls are adjacent so we need to ignore alll points that are on the walls of tetrahedrons
and all control points as those are spetial places where we allow for a point to be assigned to multiple
    tetrahedrons
"""



asdasd

# combinationss = collect(combinations(points, 3))

# to_go=true

# tetrs=[]

# t1=Meshes.Tetrahedron(sv_center
# ,combinationss[1][1]
# ,combinationss[1][2]
# ,combinationss[1][3]  )
# push!(tetrs,t1)



# while to_go
#     for i_perm in collect(enumerate(combinationss))
#         i,perm=i_perm

#         t2=Meshes.Tetrahedron(sv_center
#         ,perm[1]
#         ,perm[2]
#         ,perm[3]  )
#         interss=false

#         for t in tetrs
#             if Meshes.intersects(t, t2)
#                 interss=true
#             end
#         end

#         if !interss
#             print(t2)
#             push!(tetrs,t2)
#             deleteat!(combinationss, i)  # This will remove the element at index 3
#             to_go=true
#         end
#         push!(tetrs,t2)

#     end
# end


# tetr_1=Meshes.Tetrahedron(Meshes.Point3(1,1,1),Meshes.Point3(1,2,1),Meshes.Point3(1,1,2),Meshes.Point3(2,2,2))
# tetr_2=Meshes.Tetrahedron(Meshes.Point3(1,1,1),Meshes.Point3(1,2,1),Meshes.Point3(2,1,2),Meshes.Point3(2,2,2))

# I = Meshes.intersects(tetr_1, tetr_2)
# tetr_1 ∩ tetr_2
# viz(tetr_2)

"""
idea now that we want to construct our supervoxel volume from tetrahedrons where all tetrahedtons will have an apex
in the sv center it is non obious how to construct the tetrahedrons so they do not intersect - we will need to check it
simple solution (slow but simple) is to get random 3 points and sv center and check if it intersects with any of the tetrahedrons
that are already present in the volume do it so long as we have no longer any tetrahedron that meet our criterion

if we will have futher problems with tetrahedrons intersections we can probably deal with it by checking intersection of their walls if none 
    intersects then tetrahedron do not intersect
"""