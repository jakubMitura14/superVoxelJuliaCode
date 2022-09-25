import Lux
import NNlib, Optimisers, Plots, Random, Statistics, Zygote, HDF5

NNlib Optimisers Plots Random Statistics Zygote HDF5
## some constants
Nx, Ny, Nz = 32, 32, 32
oneSidePad = 1
totalPad = oneSidePad*2
dim_x,dim_y,dim_z= Nx+totalPad, Ny+totalPad, Nz+totalPad

crossBorderWhere = 16

numClasses= 8

##

conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())
function getConvModel()
    return Lux.Chain(conv1(1,4),conv1(4,8),conv1(8,16),conv1(16,8),conv1(8,4),conv1(4,2),conv1(2,1))
end#getConvModel

net=getConvModel()

Dense(in_dims => out_dims, activation=identity
clusterLayer=Lux.Dense(10, class_num, bias=False)

function XAIlossfunction(epoch)


end    