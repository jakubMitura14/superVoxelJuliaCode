

using MedPipe3D
using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved
# using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA
using HDF5,Colors
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using Hyperopt,Plots
using MedPipe3D.LoadFromMonai
using Lux
using Distributed
sitk=MedPipe3D.LoadFromMonai.getSimpleItkObject()

# some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl


conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1)
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=2)
