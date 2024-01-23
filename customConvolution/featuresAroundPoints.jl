"""
I)for each voxel define paths of length for example 5 in all 6 or 9 directions
    for each point in a path save the Statistics
    1) value at this point
    2) distance from origin point - taking spacing into account
    3) mean around given point in path
    4) variance of neighberhood of given point in a path 
    5) strength of edge in given point measured as a variance of intensity in the line 
        perpendicular to the path

   We will start from analysis in 6 directions - the end float that will be a result from each direction
   after parametrization will be saved in separate 3D array for later usage in diffusion     

    implementation details
    the variance from point 4 will not be true variance we will just subtract the currently analyzed value in a path minus values in surroundings
    and square the diffrences
    we will use x,y,z dimensions of the threadblock to conviently paralelize the 
    calculations in diffrent directions and diffrent features

    we will set that x dim will tell us which feature we are calculating, y which direction we are analyzing currently
    and z dim how far in this direction we travel

"""


"""
as we look for the begining in 6 diffrent directions we need a macro that will be a wrapper
    and call the supplied expression over all directions, need also to keep in consideration 
    how the threads will be utilized to their best extent
"""
macro iterateOverAllDirections()
    return  esc(quote


    end)
end#iterateOverAllDirections

"""
we will analyze the block and grid localisation and on its basis it will tell what is the voxel 
that is central for given thread - to be more precise we need to get a location at path
so we need an information about what direction we travel - hence block y and how far we are hence z
we will set convention of direction
"""
function getForThreadCenter()

end# getForThreadCenter   

"""
we willhave 3 dimensional blocks definitions and have 1 block per voxel

"""
function testKern(A, p, Aout,Nxa,Nya,Nza)
    #adding one bewcouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    shared = CuStaticSharedArray(Float32, 5,6,8)
    
    
    #1)value
    #Aout[x, y, z] = A[x, y, z] *p[x, y, z] *p[x, y, z] *p[x, y, z] 
    
    return nothing
end