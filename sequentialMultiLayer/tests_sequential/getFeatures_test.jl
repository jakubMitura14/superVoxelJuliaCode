using Revise, Test, Plots
includet("/media/jakub/NewVolume/projects/superVoxelJuliaCode/sequentialMultiLayer/utilsSequential/includeAllSequential.jl")
using Main.generate_synth_simple
#testing the model whheather it compiles

rng = Random.MersenneTwister()
Nx, Ny, Nz=64,64,64
oneSidePad=0
crossBorderWhere=32

# base_arr=rand(dim_x,dim_y,dim_z )
# base_arr=Float32.(reshape(base_arr, (dim_x,dim_y,dim_z,1,1)))
threads_CalculateFeatures=(8,8,8) 
blocks_CalculateFeatures=(8,8,8)
threads_CalculateFeatures_variance=(8,8,8)
blocks_CalculateFeatures_variance=(8,8,8)
# ps, st = Lux.setup(rng, model)
# out = Lux.apply(model, base_arr, ps, st)
# size(out[1])

######## define features array
image,indArr=createTestDataFor_Clustering(Nx, Ny, Nz, oneSidePad, crossBorderWhere)
r_features=4
r_feature_variance=3
featuresNumb=2

image_withFeature_var=prepareFeatures( CuArray(image)
    ,r_features
    ,r_feature_variance
    ,featuresNumb
    ,threads_CalculateFeatures,blocks_CalculateFeatures
    ,threads_CalculateFeatures_variance,blocks_CalculateFeatures_variance )


outCpu= Array(image_withFeature_var)
p1 = heatmap(outCpu[:,:,30,1,1]) 
p2 =  heatmap(outCpu[:,:,30,2,1]) 
p3 =  heatmap(outCpu[:,:,30,3,1]) 
p4 =  heatmap( (  (outCpu[:,:,30,3,1]./(maximum(outCpu[:,:,30,3,1])) ) + outCpu[:,:,30,2,1]./(maximum(outCpu[:,:,30,2,1]))  )./2 ) 

plot(p1, p2, p3,p4, layout = (2, 2), legend = false)


# pushfirst!(LOAD_PATH, raw"/root/.vscode-server/extensions/julialang.language-julia-1.7.12/scripts/packages");using VSCodeServer;popfirst!(LOAD_PATH);VSCodeServer.serve(raw"/tmp/vsc-jl-repl-32c620ce-8fec-4cc7-9734-a51dc9081d48"; is_dev = "DEBUG_MODE=true" in Base.ARGS, crashreporting_pipename = raw"/tmp/vsc-jl-cr-21ad6b6a-99bc-48b1-9b02-91c6db49eaba");nothing # re-establishing connection with VSCode