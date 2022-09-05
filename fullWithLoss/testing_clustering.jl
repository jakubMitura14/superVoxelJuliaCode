includet("/media/jakub/NewVolume/projects/superVoxelJuliaCode/utils/includeAll.jl")
using Distributions

Nx, Ny, Nz = 32, 32, 32
oneSidePad = 1
totalPad = oneSidePad*2
dim_x,dim_y,dim_z= Nx+totalPad, Ny+totalPad, Nz+totalPad

crossBorderWhere = 16
# sitk=MedPipe3D.LoadFromMonai.getSimpleItkObject()
pathToHDF5="/home/jakub/CTORGmini/smallDataSet.hdf5"
data_dir = "/home/jakub/CTORGmini"

#how many gaussians we will specify 
const gauss_numb_top = 8
r=3 #the radius for features calculations
featuresNumb=2 #number of analyzed features (4th dimension ...)
threads_apply_gauss = (8, 8, 8)
blocks_apply_gauss = (4, 4, 4)
threads_CalculateFeatures= (8, 8, 8)
blocks_CalculateFeatures = (4, 4, 4)


rng = Random.default_rng()
origArr,indArr=createTestDataFor_Clustering(Nx, Ny, Nz, oneSidePad, crossBorderWhere)


modelConv = getConvModel()
gaussApplyLayer=Gauss_apply(gauss_numb_top,threads_apply_gauss,blocks_apply_gauss,1)

"""
important skip connection here gets input and concatenate it with the output of the last layer
and the concatenated dimensions are last so for example if we have 3 channels input and 1 channel output of the model
    and we will concatenate
    we will have output of a model as a first channel and channels 2,3,4 will be input
"""
model=Lux.Chain(
    Lux.SkipConnection(modelConv, myCatt),gaussApplyLayer )

ps, st = Lux.setup(rng, model)
# x =reshape(origArr, (dim_x,dim_y,dim_z,1,1))
x= CuArray(origArr)
x=call_calculateFeatures(x,size(x),r,featuresNumb,threads_CalculateFeatures,blocks_CalculateFeatures )
#additionally we want to normalize the input in each layer separately
imageView=view(x,:,:,:,1,:)
meanView=view(x,:,:,:,2,:)
varView=view(x,:,:,:,3,:)
#normalization
imageView[:,:,:,:,:]= imageView./maximum(imageView) 
meanView[:,:,:,:,:]= meanView./maximum(meanView) 
varView[:,:,:,:,:]= varView./maximum(varView) 
# ps=ps.|> Lux.gpu

# y_pred, st =Lux.apply(model, x, ps, st) 

opt = Optimisers.NAdam(0.003)
#opt = Optimisers.Adam(0.003)
#opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0), Optimisers.NAdam());

function clusteringLossA(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    # print("   sizzzz $(size(y_pred))       ")
    res= sum(y_pred)
    return res, st, ()

    # return 1*(sum(y_pred)), st, ()
end


# tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
#tstate = Lux.Training.TrainState(rng, model, opt)
vjp_rule = Lux.Training.ZygoteVJP()


function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data,
    epochs::Int)
   # data = data .|> Lux.gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, clusteringLossA,
                                                                data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

tstate = main(tstate, vjp_rule, x,1)
tstate = main(tstate, vjp_rule, x,3000)

# tstate = main(tstate, vjp_rule, x,1500)



############################ visualization


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

    out[x,y,z,1,1]=float(index)    
    return nothing
end
psss=tstate.parameters
a,b=psss
stdGaus,means=b
out = CUDA.zeros(size(origArr))

# pss= Lux.gpu(psss)
# stt= Lux.gpu(st)
# y_pred, st = Lux.apply(model, origArr, pss, stt)
@cuda threads = threads_apply_gauss blocks = blocks_apply_gauss applyGaussKern_for_vis(means,stdGaus,x,out,gauss_numb_top)

outCpu= Array(out)

maximum(outCpu)
minimum(outCpu)

p1 = heatmap(outCpu[:,:,8]) 
p2 = heatmap(outCpu[:,19,:])
p3 = heatmap(outCpu[2,:,:])
p4 = heatmap(outCpu[30,:,:])

plot(p1, p2, p3, p4, layout = (2, 2), legend = false)



l6
1+1
# using Pkg
# Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
# 1+1
# heatmap(indArr[3,:,:])


# 1+1

# # cpuArr = Array(A[3, :, :])
# # heatmap(cpuArr)


# ### run
# threads = (4, 4, 4)
# blocks = (2, 2, 2)

# maxx = maximum(A)
# @cuda threads = threads blocks = blocks scaleDownKernDeffP(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)

# @cuda threads = threads blocks = blocks normalizeKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout, maxx)

# for i in 1:80
#     @cuda threads = threads blocks = blocks scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
#     #A=Aout
#     @cuda threads = threads blocks = blocks expandKernelDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)
#     A = Aout
#     maxx = maximum(A)
#     @cuda threads = threads blocks = blocks normalizeKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout, maxx)
#     # @cuda threads = (4, 4, 4) blocks = (2, 2, 2) scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)

# end
# @cuda threads = threads blocks = blocks scaleDownKernDeff(Nx, Ny, Nz, A, dA, p, dp, Aout, dAout)

# totalPad=oneSidePad*2
# Aempty = Float32.(rand(Nx+totalPad, Ny+totalPad, Nz+totalPad ) )
# dAempty= Float32.(ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) )




# anteriore=Ny+ oneSidePad

# meanDiff=3


# indexxx=0
# function setRandValues(vieww)
#     global indexxx=indexxx+1
#     sizz=size(vieww)
#     vieww[:,:,:]=rand(Normal(meanDiff*indexxx,10.0), sizz)
# end

# setRandValues(top_left_post)
# setRandValues(top_right_post)
# setRandValues(top_left_ant)
# setRandValues(bottom_left_post)
# setRandValues(bottom_right_post)
# setRandValues(bottom_left_ant)
# setRandValues(bottom_right_ant)


# # clusteringLossTest(Nx, Ny, Nz,crossBorderWhere, oneSidePad,Array(Aout), Array(p))
# #clusteringLossTest(Nx, Ny, Nz,crossBorderWhere, oneSidePad,Aempty, Array(p))
# outLossNum=Float32(0.0)
# outLoss=[outLossNum]
# doutLoss=[outLossNum]
# # clusteringLossTest(outLoss,Nx, Ny, Nz,crossBorderWhere, oneSidePad,Aempty,tops,tope, bottoms, bottome, lefts, lefte, rights, righte, anteriors,anteriore ,posteriors , posteriore)
# # outLoss
# # outLoss[1]=0.0




# aaa=clusteringLossTestDeff(outLoss,doutLoss,Nx, Ny, Nz, Aempty, dAempty, crossBorderWhere, oneSidePad,tops,tope, bottoms, bottome, lefts, lefte, rights, righte, anteriors,anteriore ,posteriors , posteriore)
# aaa
# outLoss
# outLoss[1]
# Int(round(outLoss[1]))


# var(Aempty)



# cpuArr = Array(Aout[3, :, :])
# topLeft = cpuArr[2, 2]
# topRight = cpuArr[9, 2]
# bottomLeft = cpuArr[2, 9]
# bottomRight = cpuArr[9, 9]

# topLeftCorn = Array(Aout[3, :, :])
# topRightCorn = Array(Aout[3, :, :])
# bottomLeftCorn = Array(Aout[3, :, :])
# bottomRightCorn = Array(Aout[3, :, :])


# # cpuArr = Array(Aout[3, 2:9, 2:9])

# res=Enzyme.autodiff(clusteringLossTest, Active #this tell about return
# # , Const(Nx), Const(Ny)
# # , Const(Nz), Duplicated(A, dA), Duplicated(p, dp) , Const(crossBorderWhere), Const(oneSidePad)
#  )

# res

# heatmap(cpuArr)

# print("topLeft $(topLeft) topRight $(topRight) bottomLeft $(bottomLeft)  bottomRight $(bottomRight)")



# using Revise
# using CUDA, Enzyme, Test, Plots

# function arsum(out,Nx, Ny, Nz,crossBorderWhere, oneSidePad,A, p) 
#     out[1]=11.0
#     g = 11.0
#     # for elem in f
#     #     g += elem
#     # end
#     # out[1]=g
#     return g
# end




# Nx, Ny, Nz = 8, 8, 8
# oneSidePad = 1
# crossBorderWhere = 4
# totalPad=oneSidePad*2
# Aempty = ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
# dAempty= ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 

# pempty = ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 
# dpempty= ones(Nx+totalPad, Ny+totalPad, Nz+totalPad ) 


# out=[0.0]
# dout=[0.0]
# Enzyme.autodiff(Reverse, arsum, Active,  Duplicated(out, dout), Const(Nx), Const(Ny)
# , Const(Nz), Duplicated(Aempty,dAempty), Duplicated(pempty, dpempty) , Const(crossBorderWhere), Const(oneSidePad) )

# @test inp ≈ Float64[1.0, 2.0]
# @test dinp ≈ Float64[1.0, 1.0]
# out



# const tops=1+Int(oneSidePad)
# const tope=Int(crossBorderWhere)+ Int(oneSidePad)
# const bottoms=crossBorderWhere
# const bottome=Nz+ oneSidePad
# const lefts=1+oneSidePad
# const lefte=crossBorderWhere+ oneSidePad
# const rights=crossBorderWhere+ oneSidePad+1
# const righte=Nx+ oneSidePad
# const anteriors=crossBorderWhere+ oneSidePad+1
# const anteriore=Ny+ oneSidePad
# const posteriors=1+oneSidePad
# const posteriore=crossBorderWhere+ oneSidePad


# function loss_function(model, ps, st, x)
#     y_pred, st = Lux.apply(model, x, ps, st)

#     top_left_post =view(y_pred,tops:tope,lefts:lefte, posteriors:posteriore )
#     top_right_post =view(y_pred,tops:tope,rights:righte, posteriors:posteriore)
    
#     top_left_ant =view(y_pred,tops:tope,lefts:lefte, anteriors:anteriore )
#     top_right_ant =view(y_pred,tops:tope,rights:righte, anteriors:anteriore ) 
    
#     bottom_left_post =view(y_pred,bottoms:bottome,lefts:lefte, posteriors:posteriore ) 
#     bottom_right_post =view(y_pred,bottoms:bottome,rights:righte, posteriors:posteriore ) 
    
#     bottom_left_ant =view(y_pred,bottoms:bottome,lefts:lefte, anteriors:anteriore )
#     bottom_right_ant =view(y_pred,bottoms:bottome,rights:righte, anteriors:anteriore ) 
    
#     varss= map(var  ,[top_left_post,top_right_post, top_left_ant,top_right_ant,bottom_left_post,bottom_right_post,bottom_left_ant, bottom_right_ant ])
#     means= map(mean  ,[top_left_post,top_right_post, top_left_ant,top_right_ant,bottom_left_post,bottom_right_post,bottom_left_ant, bottom_right_ant ])
    
    
#     # so we want maximize the ypred values so evrywhere we will have high prob in some gaussian
#     # minimize variance inside the regions
#     # maximize variance between regions
#     res= sum(varss)-sum(y_pred) -var(means)
#     return res, st, ()

#     # return 1*(sum(y_pred)), st, ()
# end