
using Flux,Lux, Random,Optimisers

"""
get transposed convolution from flux to Lux
# some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl
most critical idea in unet Outputs need to match !
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


conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.relu, stride=1, pad=Lux.SamePad())
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.relu, stride=2, pad=Lux.SamePad())
tran = ( in, out) -> Flux.ConvTranspose((3, 3, 3), in=>out, relu, stride=2, pad=Flux.SamePad())

tran2(in_chan,out_chan) = FluxCompatLayer(tran(in_chan,out_chan))



"""
first we want to reduce the size of the image so to encode in a smaller space all required data 
atleastby assumption 
Later good to experiment with redefining it as a unet just with shared convolutions
"""
function getContractModel(inChan,outChan)

    return Lux.Chain(conv2(inChan, 4)
                        ,conv2(4, 8)
                        ,conv2(8, 8)
                        ,conv2(8, outChan)
                        )

end #getContractModel

"""
idea is to get as much layers of this type as possible maximum arbitrary number of supervoxels
becouse of possibly large quatity one needs to be cautious of the umber of parameters required - by tis layer
-there should befew parameters
still we will start with dese layer to enable taking location into consideration
and then do transposed convolutions to regain original dimensions
"""
function getPerSVLayer(inChan,outChan,InDimX,inDimY,iDimZ)
    
    return Lux.Chain(MultLayer((InDimX,inDimY,iDimZ,inChan,1)),
                        tran2(inChan, 1)
                        ,tran2(1, 1)
                        ,tran2(1, 1)
                        ,tran2(1, outChan)
                        )
end#getPerSVLayer



"""
concatenate on 4th (channel) plus scalar 
so a is reduced representation
b is just original image with its features as channels 
concatenetes probabilities with original input and adds to tuple the spread loss
"""
function catTupleAfterSpread_loss(a,b)
   return (cat(a[1],b;dims=4),a[2],)
end  


"""
creates parts of the model in order to later compose it 
we get the series of strided convolutions that are contracting part
then point wise 3d array multiplication that is simplified dense leyer
then series of transposed convolutions to recreate original dimensions
semidense and transposed convolutions will be trained separately for each supervoxels
hence those layers will be instantiated as many times as is planned maximum supervoxel number
what is important each supervoxel layer will end with loss calculations for which the features 
    taht are input for whole model will be needed 
    becouse of it we will branch the model and concatenete resultto pass the input on

    numberOfConv2 - how many times we executet convolutions with stride 2 - we need then to do basically the same with transposed convolution
    dim_x,dim_y,dim_z - main input arr dimensions
    featureNumb - how many features we analyze   
    supervoxel_numb - how many supervoxels we want
    threads_blocks_struct - struct holding specifications of threads and blocks required for kernels used in the model
    model output is output is ((a,l ),r  ) a - concatenated array where first channel is probability map of last supervoxel layer second is added contributions from supervoxels probability maps third original array and futher channels a features variance
                                            l - is accumulated scalar loss and r is reduced representation of the array
"""
function getModel(numberOfConv2,dim_x,dim_y,dim_z, featureNumb, supervoxel_numb,threads_blocks::threads_blocks_struct )
    reductionFactor=2^numberOfConv2
    rdim_x,rdim_y,rdim_z=Int(round(dim_x/reductionFactor )),Int(round(dim_y/reductionFactor )),Int(round(dim_z/reductionFactor ))
    #featureNumb+1 becouse we also get original image as channel 1
    # we pass to the output input so
    # output will be tuple where first entry in reduced representation
    #second entry is the input - feature array
    contr=Lux.SkipConnection(getContractModel(featureNumb+1,2),get_tuple)
    layers= Vector{Lux.AbstractExplicitLayer}(undef,supervoxel_numb)

    # for first supervoxel layer the workflow will be slightly diffrent as we need to concatenate it
    sc_param_1= Parallel(myGetTuple,    
                    Parallel(myCatt4#concatenetion of Outputs
                    #first we need to process 
                    ,Lux.Chain(SelectTupl(1) # here we get just the reduced representation 
                    ,getPerSVLayer(2,1,rdim_x,rdim_y,rdim_z ) # it holds trainable parameters for this first supervoxel  output will the same first 3 dimensions as primar imput(orig array with features)
                        )
                    #we are just passing the input to be concatenated to output
                    ,NoOpLayer()
                        )
                    # this select tupl enable us pass the reduced representation down the model
                    ,SelectTupl(1)

                )#output is tuple (a,b) a- probability map of first supervoxel b- reduced representation of image 

    #so sc_param_1 will give tuple where first entry is an array with concateneted first voxel probability map and original array
    #and second tuple entry is the reduced representation
    layers[1]= Parallel(myGetTuple,
                        Parallel(myGetTuple,Lux.Chain(
                        SelectTupl(1), # here we get just the concatenated original input and supervoxel probability map     
                        sc_param_1),
                        0.0 # initializing scalar loss                        
                        )
                        ,spreadKern_layer(dim_x,dim_y,dim_z,probMapChannel,featuresStartChannel,threads_spreadKern,blocks_spreadKern)# it return both its input and scalar loss as tuple
                        ,featureLoss_kern__layer(dim_x,dim_y,dim_z,probMapChannel,featuresStartChannel,threads_featureLoss_kern_,blocks_featureLoss_kern_,featureNumb)
                        SelectTupl(2) # here we get just the reduced representation 
                        )# output is ((a,l ),r  ) a- concateneted probability map of first supervoxel and input, l - scalar loss r - scalar accumulated loss of first supervoxel

    #we start iterating over all supervoxels with exception of the first one as this is already done

    for superVoxelIndex in 2:supervoxel_numb
        #input is the nested tupleis ((a,l ),r  ) a- concateneted probability map of first supervoxel and input, l - scalar loss r - scalar accumulated loss
        sc_param= Parallel(myGetTuple,    
                        Parallel(myCatt4#concatenetion of Outputs
                        #first we need to process 
                        ,Lux.Chain(SelectTupl(2) # here we get r
                        ,getPerSVLayer(2,1,rdim_x,rdim_y,rdim_z ) # it holds trainable parameters for this first supervoxel  output will the same first 3 dimensions as primar imput(orig array with features)
                            )
                        #we are just passing the input to be concatenated to output
                        ,Lux.Chain(SelectTupl(1),SelectTupl(1))# here we get concateneted orig array 
                            )
                        # this select tupl enable us pass the reduced representation down the model
                        ,Lux.Chain(SelectTupl(1),SelectTupl(2))# here we get scalar loss

                    )#output is ((a,l ),r  ) a- concateneted probability map of this supervoxel (channel 1) previous super voxels (channel 2) and input (channels from 3 to end), l - scalar loss r - scalar accumulated loss

        #so sc_param_1 will give tuple where first entry is an array with concateneted first voxel probability map and original array
        #and second tuple entry is the reduced representation
        layers[superVoxelIndex]= Parallel(myGetTuple,
                            Parallel(myGetTuple,Lux.Chain(
                                SelectTupl(1),  
                                SelectTupl(1), # here we get just the concatenated original input and supervoxel probability map     
                                sc_param), Lux.Chain(
                                    SelectTupl(1),  
                                    SelectTupl(2) # here we get previous loss
                                    )
                             # spreadKern_layer expects tupl  (a,l ) where a is big concateneted array and l is scalar loss that is acumulated as going through layers      
                            ,spreadKern_layer(dim_x,dim_y,dim_z,probMapChannel,featuresStartChannel,threads_spreadKern,blocks_spreadKern)# it return both its input and scalar loss as tuple
                            ,featureLoss_kern__layer(dim_x,dim_y,dim_z,probMapChannel,featuresStartChannel,threads_featureLoss_kern_,blocks_featureLoss_kern_,featureNumb)
                            ,disagreeKern_layer(dim_x,dim_y,dim_z,threads_disagreeKern,blocks_disagreeKern)
                            )
                            SelectTupl(2), # here we get just the reduced representation to pass on
                            )
    end# for supervoxel_numb  

    return Lux.Chain(layers...)

end #getModelParts



# dim_x,dim_y,dim_z=32,32,32
# base_arr=rand(dim_x,dim_y,dim_z )
# # vggModel=vgg_block(1,1,2)
# rng = Random.MersenneTwister()
# base_arr=reshape(base_arr, (dim_x,dim_y,dim_z,1,1))
# #model = Lux.Chain(Lux.Dense(100, 13200, NNlib.tanh),  )

# in_chs=1
# lbl_chs=1
# # Contracting layers
# # l1 = Chain(conv1(in_chs, 4))
# # l2 = Chain(l1, conv1(4, 4), conv2(4, 16))
# # l3 = Chain(l2, conv1(16, 16), conv2(16, 32))
# # l4 = Chain(l3, conv1(32, 32), conv2(32, 64))
# # l5 = Chain(l4, conv1(64, 64), conv2(64, 128))

# # # Expanding layers
# # l6 = Chain(l5, tran2(128, 64), conv1(64, 64))
# # l7 = Chain(Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))       # Residual connection between l6 & l4
# # l8 = Chain(Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))       # Residual connection between l7 & l3
# # l9 = Chain(Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))          # Residual connection between l8 & l2
# # l10 = Chain(l9, conv1(4, lbl_chs))



# l1 = conv2(1, 1)
# l2 = conv2(1, 1)
# ll1=Lux.Chain(l1,l2)

# l3=tran2(1, 1)
# l4=tran2(1, 1)
# l5=tran2(1, 1)

# model = Lux.Parallel(+, Lux.Chain(ll1,l3), l2)

# rng = Random.MersenneTwister()
# base_arr=reshape(base_arr, (dim_x,dim_y,dim_z,1,1))

# ps, st = Lux.setup(rng, model)
# out = Lux.apply(model, base_arr, ps, st)
# size(out[1]) # we gat smaller 





