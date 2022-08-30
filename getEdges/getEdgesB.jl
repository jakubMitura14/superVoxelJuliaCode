

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


sitk=MedPipe3D.LoadFromMonai.getSimpleItkObject()

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


# some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl

using TestImages, FileIO, ImageView,ImageEdgeDetection

conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1)
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=2)


"""pdf of the univariate normal distribution."""
function univariate_normal(x, mean, variance)
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))
end#univariate_normal

# """
# good tutorial below
# https://gowrishankar.info/blog/calculus-gradient-descent-optimization-through-jacobian-matrix-for-a-gaussian-distribution/
# generally estimating mean and variance of gaussian distributions needs to be part of the backpropagation chain hence need to be 
# done via optimazation based on idea on gradient descent - challenge here is to keep the SPD structure of the covariance matrix
#     simple way to avoid it is to just train univariate distributions - and hopfully the other layesr will lead to the intensity values that are uniform enough 
# """
# function optimize_gauss_backprop()

# end

# """
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://leenashekhar.github.io/2019-01-30-KL-Divergence/
# """
# function klDiv_univariate_gaussians()
#     KL(p,q)=logσ2σ1+σ21+(μ1−μ2)22σ22−12
# end    


# """
# https://sunil-s.github.io/assets/pdfs/multivariate_mutual_information.pdf
# """
# function mutualInformation()


# end#mutualInformation

"""
loss function will look for supervoxels in data
    1) we need to define series of gaussian distributions where each will be responsible for single supervoxel
    2) we would like to minimize the amount of supervoxel in this model it would mean that we want to maximize the number of the gaussians that would have 0 corresponding voxels
    3) we want to make the distributions of the gaussians as dissimilar as possible - for example big kl divergence ? jensen inequality? 
    4) we will associate gaussian with the region if probability evaluated for it is the biggest one among all gaussians
    5) we need also to add information that he spatial variance measured as the squared sum of distances from centroids is maximized
"""
function clusteringLoss()
    #1) calculate gaussian for each gaussian pdf and save the max or pseudo max as algoOutput
    #2) sum the output from 1 - we want to maximize this sum - so all points in an image will have high probability in some distribution 
    
    #3)evaluate the similarity between distributions - for simplification for the beginning we will assume the same variance 
        #for all variables so the measure of the dissimilarity between distributions could be described by just 
        #variance of their means and we want to maximize this variance
    
    #4) for each gaussian we want it to be clustered so we want to minimize distance of high probability voxels from themselves
        #simple metric would be to reduce the variance of the calculated probbailities in nieghberhoods around each points
    
    #5) we can consider sth like relaxation labelling to strengthen the edges after all




end#clusteringLoss

