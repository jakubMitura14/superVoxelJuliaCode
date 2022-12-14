



@inline function transConvAround(x,y,z  ,new_x,new_y,new_z  ,xloc,yLoc,zLoc  ,xLoc_in,yLoc_in,zLoc_in ,paramIndex
     ,dims_input_x,dims_input_y,dims_input_z_x,   output,params,input)
    # if (y+yLoc_in>=1 && y+yLoc_in<=dims_input_y && z+zLoc_in>=1 && z+zLoc_in<= dims_input_z && x+xLoc_in>=1 && x+xLoc_in<= dims_input_x)
    #     #output[new_x,new_y,new_z]+=params[xloc+2,yLoc+2, zLoc+2,paramIndex]*input[x+xLoc_in,y+yLoc_in,z+zLoc_in]
    # end #if
end #transConvAround


#####################     kernels      #######################

"""
custom implementation of transposed convolution in order to use the Enzyme for automatic differentiation
    it will additionally store the sums of the outputs in order to enable later normalizations
    input - input 3d array its dimensions stated in dims_input
    output - output - bigger then input 3d aray its dimensions stated in dims_output
    params - 4d array which will be interpreted as list of 3d cubes of length numParams that constitute parameters for transposed convolutions
    algorithm:
       We will do dsome simplification we will enlarge the array that we have by adding first planes in x direction then in y then in z 
       each time it will lead to separate invocation of kernel
"""
function transConv_kernel_x(input,output,dims_input_x,dims_input_y,dims_input_z,params,numParams)
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z()))
    #check if we are in range
    if(x<(dims_input_x+1) && y<(dims_input_y+1) && z<(dims_input_z+1)) 
        #calculating coordinates of new center +2 becouse we have 1 based indexing
        new_x = ((x-1)*2)+1
        for paramIndex in numParams
            for yLoc in -1:1, zLoc in -1:1
                transConvAround(x,y,z  ,new_x,y,z  ,-1,yLoc,zLoc  ,0,yLoc,zLoc  ,paramIndex,dims_input_x,dims_input_y,dims_input_z,output,params,input)
                transConvAround(x,y,z  ,new_x,y,z  ,1,yLoc,zLoc  ,1,yLoc,zLoc  ,paramIndex,dims_input_x,dims_input_y,dims_input_z,output,params,input)
            end#for locs    
        end #for num params
    end#if in range    
    return nothing

end #transConv_kernel_x   

function transConv_kernel_y(input,output,dims_input_x,dims_input_y,dims_input_z,params,numParams)
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z()))
    #check if we are in range
    if(x<(dims_input_x+1) && y<(dims_input_y+1) && z<(dims_input_z+1)) 
        #calculating coordinates of new center +2 becouse we have 1 based indexing
        new_y = ((y-1)*2)+1
        for paramIndex in numParams
            for xLoc in -1:1, zLoc in -1:1
                transConvAround(x,y,z  ,x,new_y,z  ,xLoc,-1,zLoc  ,xLoc,0,zLoc  ,paramIndex,dims_input_x,dims_input_y,dims_input_z,output,params,input)
                transConvAround(x,y,z  ,x,new_y,z  ,xLoc,1,zLoc  ,xLoc,1,zLoc  ,paramIndex,dims_input_x,dims_input_y,dims_input_z,output,params,input)
            end#for locs    
        end #for num params
    end#if in range    
    return nothing

end #transConv_kernel_y   

function transConv_kernel_z(input,output,dims_input_x,dims_input_y,dims_input_z,params,numParams)
    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
    y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) 
    z = (threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z()))
    #check if we are in range
    if(x<(dims_input_x+1) && y<(dims_input_y+1) && z<(dims_input_z+1)) 
        #calculating coordinates of new center +2 becouse we have 1 based indexing
        new_z = ((z-1)*2)+1
        for paramIndex in numParams
            for xLoc in -1:1, yLoc in -1:1
                transConvAround(x,y,z  ,x,y,new_z  ,xLoc,yLoc,-1  ,xLoc,yLoc,0  ,paramIndex,dims_input_x,dims_input_y,dims_input_z,output,params,input)
                # transConvAround(x,y,z  ,x,y,new_z  ,xLoc,yLoc,1  ,xLoc,yLoc,1  ,paramIndex,dims_input_x,dims_input_y,dims_input_z,output,params,input)
            end#for locs    
        end #for num params
    end#if in range    
    return nothing

end #transConv_kernel_z   




################# enzyme definitions   ##############



# transConv_kernel_x(input,output,dims_input_x,dims_input_y,dims_input_z,dims_output,params,numParams)

function transConv_kernel_x_Deff(
    input,d_input,
    output,d_output
    ,dims_input_x
    ,dims_input_y
    ,dims_input_z
    ,params,d_params
    ,numParams
        )

    Enzyme.autodiff_deferred(transConv_kernel_x, Const,
    Duplicated(input,d_input)
    ,Duplicated(output,d_output)
    ,Const(dims_input_x)
    ,Const(dims_input_y)
    ,Const(dims_input_z)
    ,Duplicated(params,d_params)
    ,Const(numParams)
    )
    return nothing
end


function transConv_kernel_y_Deff(
    input,d_input,
    output,d_output
    ,dims_input_x
    ,dims_input_y
    ,dims_input_z
    ,params,d_params
    ,numParams
)

    Enzyme.autodiff_deferred(transConv_kernel_y, Const,
    Duplicated(input,d_input)
    ,Duplicated(output,d_output)
    ,Const(dims_input_x)
    ,Const(dims_input_y)
    ,Const(dims_input_z)
    ,Duplicated(params,d_params)
    ,Const(numParams)
    )
    return nothing
end

function transConv_kernel_z_Deff(
    input,d_input,
    output,d_output
    ,dims_input_x
    ,dims_input_y
    ,dims_input_z
    ,params,d_params
    ,numParams
)

    Enzyme.autodiff_deferred(transConv_kernel_z, Const,
    Duplicated(input,d_input)
    ,Duplicated(output,d_output)
    ,Const(dims_input_x)
    ,Const(dims_input_y)
    ,Const(dims_input_z)
    ,Duplicated(params,d_params)
    ,Const(numParams)
    )
    return nothing
end


##### calll

function call_transConv_kernel(input,dims_input_x,dims_input_y,dims_input_z,params,numParams,threads_transConv_kernel,blocks_transConv_kernel)
    dims_outputA=(dims_input_x*2,dims_input_y,dims_input_z,1,1)
    outputA = CUDA.zeros(dims_outputA) 
    @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel transConv_kernel_x(input,outputA,dims_input_x,dims_input_y,dims_input_z,params,numParams)
    dims_outputB=(dims_input_x*2,dims_input_y*2,dims_input_z,1,1)
    outputB = CUDA.zeros(dims_outputB)
    @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel transConv_kernel_y(outputA,outputB,dims_outputA[1],dims_outputA[2],dims_outputA[3],params,numParams)
    dims_outputC=(dims_input_x*2,dims_input_y*2,dims_input_z*2,1,1)
    #overwriting outputA
    outputC = CUDA.zeros(dims_outputC)
    @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel transConv_kernel_z(outputB,outputC,dims_outputB[1],dims_outputB[2],dims_outputB[3],params,numParams)
    return outputC

end

# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_transConv_kernel),input,dims_input_x,dims_input_y,dims_input_z,params,numParams,threads_transConv_kernel,blocks_transConv_kernel)    
#    print(" dims_inputtt $(dims_input)   ")
  #  outputC =call_transConv_kernel(input,dims_input_x,dims_input_y,dims_input_z,params,numParams,threads_transConv_kernel,blocks_transConv_kernel)
    #getting the results of each step
    dims_outputA=(dims_input_x*2,dims_input_y,dims_input_z,1,1)
    outputA = CUDA.zeros(dims_outputA) 
    @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel transConv_kernel_x(input,outputA,dims_input_x,dims_input_y,dims_input_z,params,numParams)
    dims_outputB=(dims_input_x*2,dims_input_y*2,dims_input_z,1,1)
    outputB = CUDA.zeros(dims_outputB)
    @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel transConv_kernel_y(outputA,outputB,dims_outputA[1],dims_outputA[2],dims_outputA[3],params,numParams)
    dims_outputC=(dims_input_x*2,dims_input_y*2,dims_input_z*2,1,1)
    #overwriting outputA
    outputC = CUDA.zeros(dims_outputC)
    @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel transConv_kernel_z(outputB,outputC,dims_outputB[1],dims_outputB[2],dims_outputB[3],params,numParams)
    

    function call_test_kernel1_pullback(dAout)
        print(" call_test_kernel1_pullback  ")
        d_params = CUDA.ones(size(params))


        
        d_input = CUDA.zeros(size(input))

        # # we are going backwards so now we have to process first z then y and lastly x

        d_output=CuArray(collect(dAout))
        @device_code_warntype @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel  transConv_kernel_z_Deff( 
                                outputB,d_input, outputC,d_output
                                ,dims_outputB[1],dims_outputB[2],dims_outputB[3]
                                ,params,d_params
                                ,numParams
        )
        d_output=copy(d_input)

        # d_input = CUDA.zeros(dims_outputA)

        # @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel  transConv_kernel_y_Deff( 
        #                         outputA,d_input, outputB,d_output
        #                         ,dims_outputB ,dims_outputC
        #                         ,params,d_params
        #                         ,numParams
        # )

        # d_output=copy(d_input)
        # d_input = CUDA.zeros(size(input))

        # @cuda threads = threads_transConv_kernel blocks = blocks_transConv_kernel  transConv_kernel_x_Deff( 
        #                         input,d_input, outputA,d_output
        #                         ,dims_outputB ,dims_outputC
        #                         ,params,d_params
        #                         ,numParams
        # )


        print( "d_input sizee $(size(d_input) ) d_params size $(size(d_params))  " )
        return NoTangent(), d_input,NoTangent(),NoTangent(),NoTangent(),d_params,NoTangent(),NoTangent(),NoTangent()
    end   
    print( " outputC from back $(size(outputC)) ")

    return outputC, call_test_kernel1_pullback

end
##############Lux definitions

#lux layers from http://lux.csail.mit.edu/dev/manual/interface/
struct transConv_str<: Lux.AbstractExplicitLayer

    numParams::Int    
    threads_transConv_kernel::Tuple{Int,Int,Int}
    blocks_transConv_kernel::Tuple{Int,Int,Int}
end

function transConv_layer(numParams,threads_transConv_kernel,blocks_transConv_kernel )
    return transConv_str(numParams,threads_transConv_kernel,blocks_transConv_kernel)
end

function Lux.initialparameters(rng::AbstractRNG, l::transConv_str)::NamedTuple
    return (params=rand(rng, Float32, (3,3,3, l.numParams)),)
end

function Lux.initialstates(::AbstractRNG, l::transConv_str)::NamedTuple
    return (numParams=l.numParams,threads_transConv_kernel=l.threads_transConv_kernel,blocks_transConv_kernel=l.blocks_transConv_kernel )
end

function (l::transConv_str)(x, ps, st::NamedTuple)
    res=call_transConv_kernel(x,size(x)[1],size(x)[2],size(x)[3],ps.params,st.numParams,st.threads_transConv_kernel,st.blocks_transConv_kernel)
    return res ,st
end









        # shmem = CuDynamicSharedArray(Float32, 3*(blockDim().x+1) + 3*(blockDim().y+1)+ 3*(blockDim().z+1))
        # shadow_shmem = CuDynamicSharedArray(Float32, 3*(blockDim().x+1) + 3*(blockDim().y+1)+ 3*(blockDim().z+1))
