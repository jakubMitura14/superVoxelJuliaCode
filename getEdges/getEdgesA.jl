

using MedPipe3D
using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved
# using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA,HDF5,Colors,ParallelStencil, ParallelStencil.FiniteDifferences3D
using MedEval3D, MedEval3D.BasicStructs, MedEval3D.MainAbstractions
using MedEval3D, MedEval3D.BasicStructs, MedEval3D.MainAbstractions,Hyperopt,Plots
using MedPipe3D.LoadFromMonai
import Lux
import NNlib, Optimisers, Plots, Random, Statistics, Zygote, HDF5
using PythonCall



pathToHDF5="/home/jakub/CTORGmini/smallDataSet.hdf5"
data_dir = "/home/jakub/CTORGmini"
function get_example_image()
    fid = h5open(pathToHDF5, "w")
    patientNum = 1#6
    patienGroupName=string(patientNum)
    listOfColorUsed= falses(18)
    train_labels = map(fileEntry-> joinpath(data_dir,"labels",fileEntry),readdir(joinpath(data_dir,"labels"); sort=true))
    train_images = map(fileEntry-> joinpath(data_dir,"volumes",fileEntry),readdir(joinpath(data_dir,"volumes"); sort=true))
    zipped= collect(zip(train_images,train_labels))
    tupl=zipped[patientNum]
    #proper loading using some utility function
    targetSpacing=(1,1,1)
    loaded = LoadFromMonai.loadBySitkromImageAndLabelPaths(tupl[1],tupl[2],targetSpacing)
    #CTORG
    #for this particular example we are intrested only in liver so we will keep only this label
    labelArr=map(entry-> UInt32(entry==1),loaded[2])
    imageArr=Float32.(loaded[1])
    gr= getGroupOrCreate(fid, patienGroupName)
    #we save loaded and trnsformed data into HDF5 to avoid doing preprocessing every time
    saveMaskBeforeVisualization(fid,patienGroupName,imageArr,"image", "CT" )
    saveMaskBeforeVisualization(fid,patienGroupName,labelArr,"labelSet", "boolLabel" )
    writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])
    #manual Modification array
    manualModif = MedEye3d.ForDisplayStructs.TextureSpec{UInt32}(# choosing number type manually to reduce memory usage
        name = "manualModif",
        isVisible=false,
        color = getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
        ,minAndMaxValue= UInt32.([0,1]) #important to keep the same number type as chosen at the bagining
        ,isEditable = true ) # we will be able to manually modify this array in a viewer

    algoVisualization = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
        name = "algoOutput",
        # we point out that we will supply multiple colors
        isContinuusMask=true,
        colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
        ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
    )

        addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,2)
        addTextSpecs[1]=manualModif
        addTextSpecs[2]=algoVisualization

    mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)
    return mainScrollDat,tupl
end #get_example_image


mainScrollDat,tupl=get_example_image()

algoOutput= getArrByName("algoOutput" ,mainScrollDat)
CT= getArrByName("image" ,mainScrollDat)
boolLabel= getArrByName("labelSet" ,mainScrollDat)
sitk=MedPipe3D.LoadFromMonai.getSimpleItkObject()

targetSpacing=(1,1,1)
image=sitk.ReadImage(tupl[1])
image=MedPipe3D.LoadFromMonai.resamplesitkImageTosize(image,targetSpacing,sitk,sitk.sitkBSpline)

canny_filter = sitk.SobelEdgeDetectionImageFilter()
#canny_filter = sitk.CannyEdgeDetectionImageFilter()
image =  sitk.Cast(image, sitk.sitkFloat32)
# canny_filter.SetLowerThreshold(50);
# canny_filter.SetVariance(15);
canny_edges = canny_filter.Execute(image)
imageArr=MedPipe3D.LoadFromMonai.permuteAndReverseFromSitk(pyconvert(Array,sitk.GetArrayFromImage(image)))
imageArr=(imageArr.-minimum(imageArr))
imageArr=imageArr./maximum(imageArr)
maximum(imageArr)
algoOutput[:,:,:]=imageArr

# some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl

using TestImages, FileIO, ImageView,ImageEdgeDetection

img =  testimage("mandril_gray")
img_edges = detect_edges(img, Canny(spatial_scale = 1.4))

conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1)
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=2)


