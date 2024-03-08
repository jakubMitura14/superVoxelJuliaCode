using Pkg
using ChainRulesCore,Zygote,CUDA,Enzyme
using KernelAbstractions
using Zygote, Lux,LuxCUDA
using Lux, Random
import NNlib, Optimisers, Plots, Random, Statistics, Zygote
using FillArrays
using LinearAlgebra
using Images,ImageFiltering
using Revise

includet("/media/jm/hddData/projects/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")

function point_info_kern_deff(tetr_dat,d_tetr_dat
                            ,out_sampled_points,d_out_sampled_points
                            ,source_arr,d_source_arr
                            ,control_points,d_control_points
                            ,sv_centers,d_sv_centers
                            ,num_base_samp_points,num_additional_samp_points,threads,blocks)


    # shared_arr = CuStaticSharedArray(Float32, (threads[0],3))
    # d_shared_arr = CuStaticSharedArray(Float32, (threads[0],3))

    Enzyme.autodiff_deferred(Reverse,testKern, Const
                            , Duplicated(tetr_dat, d_tetr_dat)
                            , Duplicated(out_sampled_points, d_out_sampled_points)
                            , Duplicated(source_arr, d_source_arr)
                            , Duplicated(control_points, d_control_points)
                            , Duplicated(sv_centers, d_sv_centers)
                            ,Const(num_base_samp_points),Const(num_additional_samp_points) ,Const(threads)  ,Const(blocks)  )
    return nothing
end


function call_point_info_kern(tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads,blocks,pad_point_info)
    # shared_arr = CuStaticSharedArray(Float32, (threads[1],3))
    # shared_arr = CuStaticSharedArray(Float32, (100,3))
    # shared_arr = CuDynamicSharedArray(Float32, (threads[1],3))
    #shmem is in bytes
    tetr_shape=size(tetr_dat)
    to_pad_tetr=CUDA.ones(pad_point_info,tetr_shape[2],tetr_shape[3])*2
    tetr_dat=vcat(tetrs,to_pad_tetr)
    @cuda threads = threads blocks = blocks point_info_kern(tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points)
    
    tetr_dat=tetr_dat[1:tetr_shape[1],:,:]

    # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
    return out_sampled_points,tetr_dat
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_point_info_kern),prim_A, A, p,Nx,threads,blocks)
    

    Aout = calltestKern(prim_A,A, p,Nx,threads,blocks)#CUDA.zeros(Nx+totalPad, Ny+totalPad, Nz+totalPad )
    function call_test_kernel1_pullback(dAout)


        dp = CUDA.ones(size(p))
        dprim_A = CUDA.ones(size(prim_A))
        dA = CUDA.ones(size(A))
        #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)
        @cuda threads = threads blocks = blocks testKernDeff(prim_A,dprim_A, A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)

        f̄ = NoTangent()
        x̄ = dA
        ȳ = dp
        
        return dprim_A,f̄, x̄, ȳ,NoTangent(),NoTangent(),NoTangent()
    end   
    return Aout, call_test_kernel1_pullback

end
