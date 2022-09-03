using Revise
using Statistics

"""
loss function will look for supervoxels in data
    1) we need to define series of gaussian distributions where each will be responsible for single supervoxel
    2) we would like to minimize the amount of supervoxel in this model it would mean that we want to maximize the number of the gaussians that would have 0 corresponding voxels
    3) we want to make the distributions of the gaussians as dissimilar as possible - for example big kl divergence ? jensen inequality? 
    4) we will associate gaussian with the region if probability evaluated for it is the biggest one among all gaussians
    5) we need also to add information that he spatial variance measured as the squared sum of distances from centroids is maximized

    imageArr - array with original image
    pImageArr - processed image where all voxels associated to each supervoxel should have very similar value
    gaussMeans - vector with values of calculated means (this are parameters)
    gausVariance - scalar describing variance - the same for all gaussians thats means are in  gaussMeans vector
   
    #1) calculate gaussian for each gaussian pdf and save the max or pseudo max as algoOutput
    #2) sum the output from 1 - we want to maximize this sum - so all points in an image will have high probability in some distribution 
        #more precisely we would like to maximize both the sum as well as the number of the gaussians with large number of points
        #so we will effectively minimize the number of clusters
        #hence gaussians with small sums of probabilities associated so for example given two new gaussians where one will have the mean at zero and other at some value that is close 
        #to mean of biggest cluster  and we want to maximize the sum of max probabilities - so you are either big or small ...

    #3)evaluate the similarity between distributions - for simplification for the beginning we will assume the same variance 
        #for all variables so the measure of the dissimilarity between distributions could be described by just 
        #variance of their means and we want to maximize this variance
    
    #4) for each gaussian we want it to be clustered so we want to minimize distance of high probability voxels from themselves
        #simple metric would be to reduce the variance of the calculated probbailities in nieghberhoods around each points
 
    #5) and most important we should have variance of chosen features in the original image among all voxels minimized
        #using weights like in 1 and 2        

    # we can consider sth like relaxation labelling to strengthen the edges after all

    for the begining we will get a setup like this
        we have original image and 2 features mean of the neighberhood and variance of the neighberhood those will be channels 2,3,4 
        channel 1 will be output of convolutional neural network
        now 
            1) we evaluate gaussians at the given point of cnn output and its neighbours
            2) we care now only to check futher if the probability associated to those points is high enough 
            3) if we have two high probabilities in a point and one of its neighbours we check weather features are simmilar
            4) what we want is that given both points have high probability associated with given gaussian 
                it should also have similar features so the diffrence between features should be small
    """
function clusteringLoss(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    #print("   sizzzz $(size(y_pred))       ")
    return sum(y_pred), st, ()

    # return 1*(sum(y_pred)), st, ()
end
