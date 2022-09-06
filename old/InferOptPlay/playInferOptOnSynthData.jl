using Revise
includet("/media/jakub/NewVolume/projects/superVoxelJuliaCode/utils/includeAll.jl")


using EmbeddedGraphs
using Distances
using Graphs
using CUDA
using ParallelStencil
using Distributions
using InferOpt
using Optimisers
using Flux
using ForwardDiff
using Pkg
import MedEye3d
import MedEye3d.ForDisplayStructs
import MedEye3d.ForDisplayStructs.TextureSpec
using ColorTypes
import MedEye3d.SegmentationDisplay
import MedEye3d.DataStructs.ThreeDimRawDat
import MedEye3d.DataStructs.DataToScrollDims
import MedEye3d.DataStructs.FullScrollableDat
import MedEye3d.ForDisplayStructs.KeyboardStruct
import MedEye3d.ForDisplayStructs.MouseStruct
import MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
import MedEye3d.OpenGLDisplayUtils
import MedEye3d.DisplayWords.textLinesFromStrings
import MedEye3d.StructsManag.getThreeDims
import MedEye3d.DisplayWords.textLinesFromStrings
using HDF5,Colors
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved
using Logging
using Main.generate_synth_simple
using Lux, NNlib, Random, Optimisers
# using Lux


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


base_arr,randGG=getRand_graph_andArr(numberOfNodes,maxDistToJoin, dim_x,dim_y,dim_z ,minConnectedDiff,maxConnectedDiff,minDist,minEdges,number_iters)
pathToHDF5="/home/jakub/projects/hdf5Data/forGraphDataSet.hdf5"

patienGroupName="3"
fid = h5open(pathToHDF5, "w")
mainScrollDat,algoOutput=saveAndVisRandGraphNoAlgoOutPutSeen(base_arr,fid,patienGroupName,dim_x,dim_y,dim_z)

gplot(randGG)

# algoOutput= getArrByName("algoOutput" ,mainScrollDat)
# algoOutput[:,:,:]=algoOutputB


########## play Flux
# some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl

# Construct the layer




"""
get transposed convolution from flux to Lux
"""

# Layer Implementation
struct FluxCompatLayer{L,I} <: Lux.AbstractExplicitLayer
    layer::L
    init_parameters::I
end

function FluxCompatLayer(flayer)
    p, re = Optimisers.destructure(flayer)
    p_ = copy(p)
    return FluxCompatLayer(re, () -> p_)
end

#adding necessary Lux functions via multiple dispatch
Lux.initialparameters(rng::AbstractRNG, l::FluxCompatLayer) = (p=l.init_parameters(),)
(f::FluxCompatLayer)(x, ps, st) = f.layer(ps.p)(x), st
#defining transposed convolution with stride 2
# tran = (stride, in, out) -> ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=Flux.SamePad())
# tran2_prim(in_chan,out_chan) = ConvTranspose((3, 3, 3), in_chan=>out_chan, stride=2)



# some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl

tran = ( in, out) -> Flux.ConvTranspose((3, 3, 3), in=>out, stride=2, pad=Flux.SamePad())

conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1)
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=2)

tran2(in_chan,out_chan) = FluxCompatLayer(tran(in_chan,out_chan))



using Lux
# vggModel=vgg_block(1,1,2)
rng = Random.MersenneTwister()
base_arr=reshape(base_arr, (dim_x,dim_y,dim_z,1,1))
#model = Lux.Chain(Lux.Dense(100, 13200, NNlib.tanh),  )

in_chs=1
lbl_chs=1
# Contracting layers
# l1 = Lux.Chain(conv1(in_chs, 4))
# l2 = Lux.Chain(l1, conv1(4, 4), conv2(4, 16))
# l3 = Lux.Chain(l2, conv1(16, 16), conv2(16, 32))
# l4 = Lux.Chain(l3, conv1(32, 32), conv2(32, 64))
# l5 = Lux.Chain(l4, conv1(64, 64), conv2(64, 128))

# # Expanding layers
# l6 = Lux.Chain(l5, tran2(128, 64), conv1(64, 64))
# l7 = Lux.Chain(Lux.Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))       # Residual connection between l6 & l4
# l8 = Lux.Chain(Lux.Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))       # Residual connection between l7 & l3
# l9 = Lux.Chain(Lux.Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))          # Residual connection between l8 & l2
# model = Lux.Chain(l9, conv1(4, lbl_chs))


l1 = Lux.Chain(conv2(1, 1))
l2=tran2(1, 1)
mm=Lux.Chain(l1,l2)


model = Lux.Chain(
    l1, ##63,63,61
    #,Lux.Parallel(+, l1, l2)
    # tran2(lbl_chs, lbl_chs)
    # ,Lux.Conv((3,3,3),  1 => 1 , NNlib.tanh, stride=1,pad=2)
     )
#model = Lux.Parallel(+, mm, l1)

#model = Lux.Chain(l1, tran2(4, lbl_chs))


rng = Random.MersenneTwister()
base_arr=reshape(base_arr, (dim_x,dim_y,dim_z,1,1))

ps, st = Lux.setup(rng, model)
out = Lux.apply(model, base_arr, ps, st)
size(out[1]) # we gat smaller 


using Lux
# vggModel=vgg_block(1,1,2)
rng = Random.MersenneTwister()
base_arr=reshape(base_arr, (dim_x,dim_y,dim_z,1,1))
#model = Lux.Chain(Lux.Dense(100, 13200, NNlib.tanh),  )
model = Lux.Chain(
    Lux.Conv((3,3,3),  1 => 2 , NNlib.tanh, stride=2)
    ,Lux.Conv((3,3,3),  2 => 2 , NNlib.tanh, stride=2)
    ,Lux.Conv((3,3,3),  2 => 2 , NNlib.tanh, stride=2)
    ,Lux.Conv((3,3,3),  2 => 2 , NNlib.tanh, stride=2)
    ,tranConvLux(2,2)
    ,tranConvLux(2,2)
    ,tranConvLux(2,2)
    ,tranConvLux(2,2)
    ,Lux.Conv((3,3,3),  2 => 1 , NNlib.tanh, stride=1, pad=1)
    )
ps, st = Lux.setup(rng, model)
out = Lux.apply(model, base_arr, ps, st)
size(out[1]) # we gat smaller 






arr= ones(5,5,5)
isToContinue=true
while (true)
    for i in 2:4, j in 2:4, k in 2:4
        arr[i,j,j]=arr[i+1,j,k]+arr[i,j+1,k]+arr[i,j,k+1] 
    end#for
    sum(arr)>5*5*5*2 || break
end#while

arr[1]






"""
adapted from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/unet3D.jl
to Lux
"""
function unet3D(in_chs, lbl_chs)
    # Contracting layers
    l1 = Chain(conv1(in_chs, 4))
    l2 = Chain(l1, conv1(4, 4), conv2(4, 16))
    l3 = Chain(l2, conv1(16, 16), conv2(16, 32))
    l4 = Chain(l3, conv1(32, 32), conv2(32, 64))
    l5 = Chain(l4, conv1(64, 64), conv2(64, 128))

    # Expanding layers
    l6 = Chain(l5, tran2(128, 64), conv1(64, 64))
    l7 = Chain(Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))       # Residual connection between l6 & l4
    l8 = Chain(Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))       # Residual connection between l7 & l3
    l9 = Chain(Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))          # Residual connection between l8 & l2
    l10 = Chain(l9, conv1(4, lbl_chs))
end



# model = Lux.Parallel(+, Lux.Conv((1, 1,1), 3 => 3, NNlib.relu), Lux.Conv((1,1, 1), 3 => 3))




model = Lux.Chain(Lux.BatchNorm((dim_x,dim_y,dim_z)), Lux.Dense((dim_x,dim_y,dim_z), (dim_x,dim_y,dim_z,256), NNlib.tanh))


# model = Chain(BatchNorm((dim_x,dim_y,dim_z)), Dense((dim_x,dim_y,dim_z), (dim_x,dim_y,dim_z,256), NNlib.tanh)
# , BatchNorm(256),
#             Chain(Dense(256, 1, tanh), Dense(1, 10)))

# Parameter and State Variables
ps, st = Lux.setup(rng, model) .|> gpu

# Dummy Input
x = rand(rng, Float32, 128, 2) |> gpu

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
## Pullback API to capture change in state
(l, st_), pb = pullback(p -> Lux.apply(model, x, p, st), ps)
gs = pb((one.(l), nothing))[1]

# Optimization
st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)







