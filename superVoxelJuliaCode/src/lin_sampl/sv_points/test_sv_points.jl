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

dims=(2,2,2)
radius=3.0

example_set_of_svs=initialize_centers_and_control_points(dims,radius)
sv_centers,lin_x,lin_y,lin_z,oblique=example_set_of_svs

sv_center=Meshes.Point3(sv_centers[1,1,1,:])
lin_x_point_pre=Meshes.Point3(lin_x[1,2,2,:])
lin_x_point_post=Meshes.Point3(lin_x[2,2,2,:])

lin_y_point_pre=Meshes.Point3(lin_y[2,1,2,:])
lin_y_point_post=Meshes.Point3(lin_y[2,2,2,:])

lin_z_point_pre=Meshes.Point3(lin_z[2,2,1,:])
lin_z_point_post=Meshes.Point3(lin_z[2,2,2,:])

oblique_point_1=Meshes.Point3(oblique[1,1,1,:])
oblique_point_2=Meshes.Point3(oblique[2,1,1,:])
oblique_point_3=Meshes.Point3(oblique[1,2,1,:])
oblique_point_4=Meshes.Point3(oblique[1,1,2,:])
oblique_point_5=Meshes.Point3(oblique[1,2,2,:])
oblique_point_6=Meshes.Point3(oblique[2,2,1,:])
oblique_point_7=Meshes.Point3(oblique[2,1,2,:])
oblique_point_8=Meshes.Point3(oblique[2,2,2,:])
#sv_center,
points=[lin_x_point_pre,lin_x_point_post
,lin_y_point_pre,lin_y_point_post
,lin_z_point_pre,lin_z_point_post
,oblique_point_1,oblique_point_2,oblique_point_3,oblique_point_4
,oblique_point_5,oblique_point_6,oblique_point_7,oblique_point_8]

# we will create a bunch of tetrahedrons on the basis of the points above


tetrs=[
    Meshes.Tetrahedron(sv_center,lin_x_point_pre,oblique_point_1,lin_y_point_pre)
    ,Meshes.Tetrahedron(sv_center,lin_x_point_pre,oblique_point_1,lin_z_point_pre)
    ,Meshes.Tetrahedron(sv_center,lin_z_point_pre,oblique_point_1,lin_y_point_pre)

    ,Meshes.Tetrahedron(sv_center,lin_x_point_post,oblique_point_2,lin_y_point_pre)
    ,Meshes.Tetrahedron(sv_center,lin_x_point_post,oblique_point_2,lin_z_point_pre)
    ,Meshes.Tetrahedron(sv_center,lin_y_point_pre,oblique_point_2,lin_z_point_pre)

    ,Meshes.Tetrahedron(sv_center,lin_x_point_pre,oblique_point_3,lin_y_point_post)
    ,Meshes.Tetrahedron(sv_center,lin_x_point_pre,oblique_point_3,lin_z_point_pre)
    ,Meshes.Tetrahedron(sv_center,lin_z_point_pre,oblique_point_3,lin_y_point_post)

    ,Meshes.Tetrahedron(sv_center,lin_x_point_pre,oblique_point_4,lin_y_point_pre)
    ,Meshes.Tetrahedron(sv_center,lin_x_point_pre,oblique_point_4,lin_z_point_post)
    ,Meshes.Tetrahedron(sv_center,lin_z_point_post,oblique_point_4,lin_y_point_pre)


    ,Meshes.Tetrahedron(sv_center,lin_x_point_pre,oblique_point_5,lin_y_point_post)
    ,Meshes.Tetrahedron(sv_center,lin_x_point_pre,oblique_point_5,lin_z_point_post)
    ,Meshes.Tetrahedron(sv_center,lin_z_point_post,oblique_point_5,lin_y_point_post)

    ,Meshes.Tetrahedron(sv_center,lin_x_point_post,oblique_point_6,lin_y_point_post)
    ,Meshes.Tetrahedron(sv_center,lin_x_point_post,oblique_point_6,lin_z_point_pre)
    ,Meshes.Tetrahedron(sv_center,lin_z_point_pre,oblique_point_6,lin_y_point_post)

    ,Meshes.Tetrahedron(sv_center,lin_x_point_post,oblique_point_7,lin_y_point_pre)
    ,Meshes.Tetrahedron(sv_center,lin_x_point_post,oblique_point_7,lin_z_point_post)   
    ,Meshes.Tetrahedron(sv_center,lin_z_point_post,oblique_point_7,lin_y_point_pre)

    ,Meshes.Tetrahedron(sv_center,lin_x_point_post,oblique_point_8,lin_y_point_post)
    ,Meshes.Tetrahedron(sv_center,lin_x_point_post,oblique_point_8,lin_z_point_post)
    ,Meshes.Tetrahedron(sv_center,lin_z_point_post,oblique_point_8,lin_y_point_post)
    


]




viz(tetrs, color = 1:length(tetrs))


sv_center ∈ tetrs[5]




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