"""
we start from list of indicator functions fsv() - each will be associated with single supervoxel
    there may be situation that some functions do not point to any voxels in that case it means that there are less supervoxels than functions
 
I) we run kernel that return index of the function that gives highest value for given voxel
II) possibly run some postprocessing like relaxation labelling to reduce noise
III) instantiate matrix or n dimensional array where each entry in first dimension will mark one supervoxel and will hold caclulated features of this supervoxel
IV) based on physical closeness connect neighbouring supervoxels using edges
V) optionally based on features similarity add feature similarity value as a feature of the edge   
"""

using Revise
using CUDA, Enzyme, Test, Plots, HDF5

includet("/media/jakub/NewVolume/projects/superVoxelJuliaCode/fullWithLoss/utils.jl")
includet("/media/jakub/NewVolume/projects/superVoxelJuliaCode/get_synth_data/generate_synth_simple.jl")
using Main.generate_synth_simple


##some constants defining possible discrepancies between distributions defined in termes of js divergence
#for two connected distributions
minConnectedDiff=5.0
maxConnectedDiff=5.5

###parameters
#how many nodes will create a random graph
numberOfNodes=25
#maximum distance between 2 verticies that will lead to creation of the edge between them
maxDistToJoin=0.35
#dimensions of the array in which the image will be EmbeddedGraphs
dim_x,dim_y,dim_z =128,128,128
#minimum allowed distance between two nodes must be number between 0 and 1 
minDist=0.2
#minimum number of edges
minEdges=13
#set how big at the maximum should be spheres occupied by a node
number_iters = Int(round(minDist*minimum([dim_x,dim_y,dim_z])))

base_arr,randGG ,coords, gausses,edgeList =getRand_graph_andArr(numberOfNodes,maxDistToJoin, dim_x,dim_y,dim_z ,minConnectedDiff,maxConnectedDiff,minDist,minEdges,number_iters)
pathToHDF5="/home/jakub/projects/hdf5Data/forGraphDataSet.hdf5"

patienGroupName="3"
fid = h5open(pathToHDF5, "w")
mainScrollDat,algoOutput=saveAndVisRandGraphNoAlgoOutPutSeen(base_arr,fid,patienGroupName,dim_x,dim_y,dim_z)

function applyGaussKern_for_vis(means,stdGaus,origArr,out,meansLength)
    #adding one becouse of padding
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) + 1
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) + 1
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) + 1
    #iterate over all gauss parameters
    maxx = 0.0
    index=0    
    for i in 1:meansLength
       vall=univariate_normal(origArr[x,y,z,1,1], means[i], stdGaus[i]^2)
       #CUDA.@cuprint "vall $(vall) i $(i)   " 
       if(vall>maxx)
            maxx=vall
            index=i
        end #if     
    end #for

    out[x,y,z]=float(index)    
    return nothing
end


gausses