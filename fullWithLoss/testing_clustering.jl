includet("/media/jakub/NewVolume/projects/superVoxelJuliaCode/utils/includeAll.jl")
using Distributions

Nx, Ny, Nz = 32, 32, 32
oneSidePad = 1
crossBorderWhere = 16
sitk=MedPipe3D.LoadFromMonai.getSimpleItkObject()
pathToHDF5="/home/jakub/CTORGmini/smallDataSet.hdf5"
data_dir = "/home/jakub/CTORGmini"

#how many gaussians we will specify 
const gauss_numb_top = 8
threads_apply_gauss = (4, 4, 4)
blocks_apply_gauss = (2, 2, 2)



rng = Random.default_rng()


origArr,indArr=createTestDataFor_Clustering(Nx, Ny, Nz, oneSidePad, crossBorderWhere)
modelConv = getConvModel()
gaussApplyLayer=Gauss_apply(gauss_numb_top,threads_apply_gauss,blocks_apply_gauss)



ps, st = Lux.setup(rng, modelConv)
x = randn(rng, Float32, dim_x,dim_y,dim_z)
x =reshape(x, (dim_x,dim_y,dim_z,1,1))
y_pred, st =Lux.apply(modelConv, x, ps, st) 
size(y_pred)





heatmap(indArr[3,:,:])


1+1

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