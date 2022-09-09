using Flux,Lux, Random,Optimisers,Revise, CUDA
include("C:\\projects\\superVoxelJuliaCode\\differentiableClustering\\sequentialMultiLayer\\unetLux.jl")
include("C:\\projects\\superVoxelJuliaCode\\differentiableClustering\\sequentialMultiLayer\\multLayer.jl")



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



rng = Random.MersenneTwister()
dim_x,dim_y,dim_z=64,64,64
base_arr=rand(dim_x,dim_y,dim_z )
base_arr=Float32.(reshape(base_arr, (dim_x,dim_y,dim_z,1,1)))
numberOfConv2=4
reductionFactor=2^numberOfConv2
rdim_x,rdim_y,rdim_z=Int(round(dim_x/reductionFactor )),Int(round(dim_y/reductionFactor )),Int(round(dim_z/reductionFactor ))
rdim_x*rdim_y*rdim_z*2

contr=getContractModel(1,2)
sv1= getPerSVLayer(2,3,rdim_x,rdim_y,rdim_z )

model=Lux.Chain(contr,sv1 )

ps, st = Lux.setup(rng, model)
out = Lux.apply(model, base_arr, ps, st)
size(out[1])










l3=tran2(1, 1)
l4=tran2(1, 1)
l5=tran2(1, 1)



dim_x,dim_y,dim_z=8,8,8
base_arr=rand(dim_x,dim_y,dim_z )
base_arr=Float32.(reshape(base_arr, (dim_x,dim_y,dim_z,1,1)))
inLen=dim_x*dim_y*dim_z
ll=Lux.Chain(Lux,FlattenLayer(),Lux.Dense(inLen, inLen, relu),Lux.ReshapeLayer((dim_x,dim_y,dim_z,1)))
ps, st = Lux.setup(rng, ll)
out = Lux.apply(ll, base_arr, ps, st)
# model = Lux.Parallel(+, Lux.Chain(ll1,l3), l2)

a=ones(2,2,2).*2
b=ones(2,2,2).*2
a[1,1,1]=0
a.*b