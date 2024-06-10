using KernelAbstractions
using ChainRulesCore, Zygote, CUDA, Enzyme, Test
using LLVMLoopInfo



"""
calculate distance between set location and neighbouring voxel coordinates
"""
macro get_dist(a,b,c)
  return  esc(:(
  (1/((((shared_arr[threadIdx().x,1]-round(shared_arr[threadIdx().x,1]+($a) ))^2+(shared_arr[threadIdx().x,2]-round(shared_arr[threadIdx().x,2] +($b)))^2+(shared_arr[threadIdx().x,3]-round(shared_arr[threadIdx().x,3]+($c)))^2)^2  )+0.00000001))#we add a small number to avoid division by 0
))
end


"""
get a diffrence between coordinates in given axis of sv center and triangle center
"""
macro get_diff_on_line_sv_tetr(coord_i,tetr_dat_coord,point_num)
  return  esc(quote
      ((tetr_dat[$tetr_dat_coord,$coord_i]-tetr_dat[1,$coord_i])*($point_num/(num_base_samp_points+1)))
end)
end



"""
unrolled version of trilinear interpolation
"""
function unrolled_trilin_interpol(shared_arr,source_arr)
  return (((
    source_arr[Int(floor( shared_arr[threadIdx().x,1])), Int(floor( shared_arr[threadIdx().x,2])), Int(floor( shared_arr[threadIdx().x,3]))] * (1 - (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1])))) +
    source_arr[Int(ceil( shared_arr[threadIdx().x,1])), Int(floor( shared_arr[threadIdx().x,2])), Int(floor( shared_arr[threadIdx().x,3]))] * (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1])))
    )
      *
      (1 - (shared_arr[threadIdx().x,2] - Int(floor( shared_arr[threadIdx().x,2])))) +
      (source_arr[Int(floor( shared_arr[threadIdx().x,1])), Int(ceil( shared_arr[threadIdx().x,2])), Int(floor( shared_arr[threadIdx().x,3]))] * (1 - (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
       +
       source_arr[Int(ceil( shared_arr[threadIdx().x,1])), Int(ceil( shared_arr[threadIdx().x,2])), Int(floor( shared_arr[threadIdx().x,3]))] * (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
      *
      (shared_arr[threadIdx().x,2] - Int(floor( shared_arr[threadIdx().x,2]))))
     *
     (1 - (shared_arr[threadIdx().x,3] - Int(floor( shared_arr[threadIdx().x,3]))))
     +
     ((source_arr[Int(floor( shared_arr[threadIdx().x,1])), Int(floor( shared_arr[threadIdx().x,2])), Int(ceil( shared_arr[threadIdx().x,3]))] * (1 - (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
       +
       source_arr[Int(ceil( shared_arr[threadIdx().x,1])), Int(floor( shared_arr[threadIdx().x,2])), Int(ceil( shared_arr[threadIdx().x,3]))] * (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
      *
      (1 - (shared_arr[threadIdx().x,2] - Int(floor( shared_arr[threadIdx().x,2])))) +
      (source_arr[Int(floor( shared_arr[threadIdx().x,1])), Int(ceil( shared_arr[threadIdx().x,2])), Int(ceil( shared_arr[threadIdx().x,3]))] * (1 - (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
       +
       source_arr[Int(ceil( shared_arr[threadIdx().x,1])), Int(ceil( shared_arr[threadIdx().x,2])), Int(ceil( shared_arr[threadIdx().x,3]))] * (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
      *
      (shared_arr[threadIdx().x,2] - Int(floor( shared_arr[threadIdx().x,2]))))
     *
     (shared_arr[threadIdx().x,3] - Int(floor( shared_arr[threadIdx().x,3]))))
end

"""
unrolled version of calculation of local variance based on trilinear interpolation
"""
function unrolled_trilin_variance(shared_arr,source_arr)
  #first saving meaking mean 
  shared_arr[threadIdx().x,4]= unrolled_trilin_interpol(shared_arr,source_arr)
  return (((
    ((source_arr[Int(floor( shared_arr[threadIdx().x,1])), Int(floor( shared_arr[threadIdx().x,2])), Int(floor( shared_arr[threadIdx().x,3]))]-shared_arr[threadIdx().x,4])^2)
    * (1 - (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1])))) +
    ((source_arr[Int(ceil( shared_arr[threadIdx().x,1])), Int(floor( shared_arr[threadIdx().x,2])), Int(floor( shared_arr[threadIdx().x,3]))] -shared_arr[threadIdx().x,4])^2)
    * (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1])))
    )
      *
      (1 - (shared_arr[threadIdx().x,2] - Int(floor( shared_arr[threadIdx().x,2])))) +
      (((source_arr[Int(floor( shared_arr[threadIdx().x,1])), Int(ceil( shared_arr[threadIdx().x,2])), Int(floor( shared_arr[threadIdx().x,3]))] -shared_arr[threadIdx().x,4])^2)
      * (1 - (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
       +
       ((source_arr[Int(ceil( shared_arr[threadIdx().x,1])), Int(ceil( shared_arr[threadIdx().x,2])), Int(floor( shared_arr[threadIdx().x,3]))] -shared_arr[threadIdx().x,4])^2)
       * (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
      *
      (shared_arr[threadIdx().x,2] - Int(floor( shared_arr[threadIdx().x,2]))))
     *
     (1 - (shared_arr[threadIdx().x,3] - Int(floor( shared_arr[threadIdx().x,3]))))
     +
     ((  ((source_arr[Int(floor( shared_arr[threadIdx().x,1])), Int(floor( shared_arr[threadIdx().x,2])), Int(ceil( shared_arr[threadIdx().x,3]))] -shared_arr[threadIdx().x,4])^2)
     * (1 - (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
       +
       ((source_arr[Int(ceil( shared_arr[threadIdx().x,1])), Int(floor( shared_arr[threadIdx().x,2])), Int(ceil( shared_arr[threadIdx().x,3]))] -shared_arr[threadIdx().x,4])^2)
       * (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
      *
      (1 - (shared_arr[threadIdx().x,2] - Int(floor( shared_arr[threadIdx().x,2])))) +
      (  ((source_arr[Int(floor( shared_arr[threadIdx().x,1])), Int(ceil( shared_arr[threadIdx().x,2])), Int(ceil( shared_arr[threadIdx().x,3]))] -shared_arr[threadIdx().x,4])^2)
      * (1 - (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
       +
       (( source_arr[Int(ceil( shared_arr[threadIdx().x,1])), Int(ceil( shared_arr[threadIdx().x,2])), Int(ceil( shared_arr[threadIdx().x,3]))] -shared_arr[threadIdx().x,4])^2)
       * (shared_arr[threadIdx().x,1] - Int(floor( shared_arr[threadIdx().x,1]))))
      *
      (shared_arr[threadIdx().x,2] - Int(floor( shared_arr[threadIdx().x,2]))))
     *
     (shared_arr[threadIdx().x,3] - Int(floor( shared_arr[threadIdx().x,3]))))
end


"""
given point p1 and line l where line l is defined by two points p2 and p3
we calculate the distance between point p1 and line l
"""
function dist_of_p_to_line(p1,p2,p3)
  return sqrt(((((p2[2]-p3[2])*(p2[3]-p1[3]) - (p2[3]-p3[3])*(p2[2]-p1[2]))^2)+
 (((p2[3]-p3[3])*(p2[1]-p1[1]) - (p2[1]-p3[1])*(p2[3]-p1[3]))^2)+
 ((p2[1]-p3[1])*(p2[2]-p1[2]) - (p2[2]-p3[2])*(p2[1]-p1[1]))^2)) / sqrt((p3[2]-p2[2])^2+(p3[3]-p2[3])^2+(p3[1]-p2[1])^2)
end



function point_info_kern_forward(tetr_dat,out_sampled_points,source_arr,num_base_samp_points,num_additional_samp_points)
  shared_arr = CuStaticSharedArray(Float32, (256,4))
  index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))

  
  # we iterate over rest of points in main sample points
  @loopinfo unroll for point_num in UInt8(1):UInt8(num_base_samp_points)

      #we get the diffrence between the sv center and the triangle center
      shared_arr[threadIdx().x,1]= ((tetr_dat[index,5,1]-tetr_dat[index,1,1])/(num_base_samp_points+1))
      shared_arr[threadIdx().x,2]=((tetr_dat[index,5,2]-tetr_dat[index,1,2])/(num_base_samp_points+1))
      shared_arr[threadIdx().x,3]=((tetr_dat[index,5,3]-tetr_dat[index,1,3])/(num_base_samp_points+1))



      ##calculate weight of the point
      #first distance from next and previous point on the line between sv center and triangle center
      shared_arr[threadIdx().x,4]=sqrt((shared_arr[threadIdx().x,1])^2 +(shared_arr[threadIdx().x,2])^2+(shared_arr[threadIdx().x,3])^2)*2 #distance between main sample points (two times for distance to previous and next)
      #now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
      # we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
      #now we get the location of sample point

      shared_arr[threadIdx().x,1]= tetr_dat[index,1,1]+(shared_arr[threadIdx().x,1]*point_num)
      shared_arr[threadIdx().x,2]= tetr_dat[index,1,2]+(shared_arr[threadIdx().x,2]*point_num)
      shared_arr[threadIdx().x,3]= tetr_dat[index,1,3]+(shared_arr[threadIdx().x,3]*point_num)
      for triangle_corner_num in UInt8(1):UInt8(3)
        #if it is last point from base sample points we need to measure its distance to the first additional sample points not to the lines between sv center and triangle corners
        if(point_num==num_base_samp_points)
          shared_arr[threadIdx().x,4]+=sqrt( ((tetr_dat[index,triangle_corner_num+1,1]-shared_arr[threadIdx().x,1])
                  *(1/(num_additional_samp_points+1)))^2
                  +((tetr_dat[index,triangle_corner_num+1,2]-shared_arr[threadIdx().x,2])
                  *(1/(num_additional_samp_points+1)))^2
                  +((tetr_dat[index,triangle_corner_num+1,3]-shared_arr[threadIdx().x,3])
                  *(1/(num_additional_samp_points+1)))^2)
        else  
          #distance to the line between sv center and the one of triangle corners (triangle that is a base of tetrahedron)
          #in all cases line start as sv center and end as triangle corner and a point is current point in shared memory
          
          shared_arr[threadIdx().x,4]+=sqrt(((((tetr_dat[index,1,2]-tetr_dat[index,triangle_corner_num+1,2])*(tetr_dat[index,1,3]-shared_arr[threadIdx().x,3]) - (tetr_dat[index,1,3]-tetr_dat[index,triangle_corner_num+1,3])*(tetr_dat[index,1,2]-shared_arr[threadIdx().x,2]))^2)+
              (((tetr_dat[index,1,3]-tetr_dat[index,triangle_corner_num+1,3])*(tetr_dat[index,1,1]-shared_arr[threadIdx().x,1]) - (tetr_dat[index,1,1]-tetr_dat[index,triangle_corner_num+1,1])*(tetr_dat[index,1,3]-shared_arr[threadIdx().x,3]))^2)+
              ((tetr_dat[index,1,1]-tetr_dat[index,triangle_corner_num+1,1])*(tetr_dat[index,1,2]-shared_arr[threadIdx().x,2]) - (tetr_dat[index,1,2]-tetr_dat[index,triangle_corner_num+1,2])*(tetr_dat[index,1,1]-shared_arr[threadIdx().x,1]))^2)) / sqrt((tetr_dat[index,triangle_corner_num+1,2]-tetr_dat[index,1,2])^2+(tetr_dat[index,triangle_corner_num+1,3]-tetr_dat[index,1,3])^2+(tetr_dat[index,triangle_corner_num+1,1]-tetr_dat[index,1,1])^2)
        end#if else point_num==num_base_samp_points  
      end#for triangle_corner_num   
    

      #now as we had looked into distance to other points in 5 directions we divide by 5 and save it to the out_sampled_points
      # out_sampled_points[index,point_num,2]= ((shared_arr[threadIdx().x,4]/5)^3)
      out_sampled_points[index,point_num,2]= (((shared_arr[threadIdx().x,4])/5)^3)

    
      ##time to get value by interpolation and save it to the out_sampled_points

      #performing interpolation result is in var2 and it get data from shared_arr
      #saving the result of interpolated value to the out_sampled_points
      out_sampled_points[index,point_num,1]=unrolled_trilin_interpol(shared_arr,source_arr)
      #saving sample points coordinates mainly for debugging and visualizations
      out_sampled_points[index,point_num,3]=shared_arr[threadIdx().x,1]
      out_sampled_points[index,point_num,4]=shared_arr[threadIdx().x,2]
      out_sampled_points[index,point_num,5]=shared_arr[threadIdx().x,3]




  end#for num_base_samp_points

  ##### now we need to calculate the additional sample points that are branching out from the last main sample point
  @loopinfo unroll for n_add_samp in UInt8(1):UInt8(num_additional_samp_points)
    @loopinfo unroll for triangle_corner_num in UInt8(1):UInt8(3)

          #now we need to get diffrence between the last main sample point and the triangle corner
          shared_arr[threadIdx().x,1]=((tetr_dat[index,triangle_corner_num+1,1]-out_sampled_points[index,num_base_samp_points,3])/(num_additional_samp_points+1))
          shared_arr[threadIdx().x,2]=((tetr_dat[index,triangle_corner_num+1,2]-out_sampled_points[index,num_base_samp_points,4])/(num_additional_samp_points+1))
          shared_arr[threadIdx().x,3]=((tetr_dat[index,triangle_corner_num+1,3]-out_sampled_points[index,num_base_samp_points,5])/(num_additional_samp_points+1))

          #### calculate weight of the point 
          #first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
          shared_arr[threadIdx().x,4]=sqrt( shared_arr[threadIdx().x,1]^2+shared_arr[threadIdx().x,2]^2+shared_arr[threadIdx().x,3]^2)*2
          #now we get the location of sample point
          shared_arr[threadIdx().x,1]= out_sampled_points[index,num_base_samp_points,3]+(shared_arr[threadIdx().x,1]*(n_add_samp))
          shared_arr[threadIdx().x,2]= out_sampled_points[index,num_base_samp_points,4]+(shared_arr[threadIdx().x,2]*(n_add_samp))
          shared_arr[threadIdx().x,3]= out_sampled_points[index,num_base_samp_points,5]+(shared_arr[threadIdx().x,3]*(n_add_samp))
          ### now we get the distance of a point to lines joining the corner toward which current line is going and all other verticies of tetrahedron
          for tetr_dat_index in UInt8(1):UInt8(4)
            if(tetr_dat_index!=(triangle_corner_num+1) ) #this is the corner we are going to in the line for additional sampling points
              # shared_arr[threadIdx().x,4]+=(tetr_dat[index,triangle_corner_num+1,2]-tetr_dat[index,tetr_dat_index,2]- shared_arr[threadIdx().x,2])
              shared_arr[threadIdx().x,4]+=sqrt(((((tetr_dat[index,triangle_corner_num+1,2]-tetr_dat[index,tetr_dat_index,2])*(tetr_dat[index,triangle_corner_num+1,3]-shared_arr[threadIdx().x,3]) - (tetr_dat[index,triangle_corner_num+1,3]-tetr_dat[index,tetr_dat_index,3])*(tetr_dat[index,triangle_corner_num+1,2]-shared_arr[threadIdx().x,2]))^2)+
                (((tetr_dat[index,triangle_corner_num+1,3]-tetr_dat[index,tetr_dat_index,3])*(tetr_dat[index,triangle_corner_num+1,1]-shared_arr[threadIdx().x,1]) - (tetr_dat[index,triangle_corner_num+1,1]-tetr_dat[index,tetr_dat_index,1])*(tetr_dat[index,triangle_corner_num+1,3]-shared_arr[threadIdx().x,3]))^2)+
                ((tetr_dat[index,triangle_corner_num+1,1]-tetr_dat[index,tetr_dat_index,1])*(tetr_dat[index,triangle_corner_num+1,2]-shared_arr[threadIdx().x,2]) - (tetr_dat[index,triangle_corner_num+1,2]-tetr_dat[index,tetr_dat_index,2])*(tetr_dat[index,triangle_corner_num+1,1]-shared_arr[threadIdx().x,1]))^2)) / sqrt((tetr_dat[index,tetr_dat_index,2]-tetr_dat[index,triangle_corner_num+1,2])^2+(tetr_dat[index,tetr_dat_index,3]-tetr_dat[index,triangle_corner_num+1,3])^2+(tetr_dat[index,tetr_dat_index,1]-tetr_dat[index,triangle_corner_num+1,1])^2)
            end #if
          end#for

          out_sampled_points[index,(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,2]=(((shared_arr[threadIdx().x,4])/5)^3)
         
         



          #performing interpolation result is in var2 and it get data from shared_arr
          #saving the result of interpolated value to the out_sampled_points
          out_sampled_points[index,(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,1]=unrolled_trilin_interpol(shared_arr,source_arr)
          # #saving sample points coordinates
          out_sampled_points[index,(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,3]=shared_arr[threadIdx().x,1]
          out_sampled_points[index,(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,4]=shared_arr[threadIdx().x,2]
          out_sampled_points[index,(num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3,5]=shared_arr[threadIdx().x,3]







      end #for triangle_corner_num
  end #for n_add_samp


  return nothing
end





function point_info_kern_deff(tetr_dat, d_tetr_dat, out_sampled_points, d_out_sampled_points, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, num_base_samp_points, num_additional_samp_points
    )

        Enzyme.autodiff_deferred(Enzyme.Reverse, point_info_kern_forward, Const
            , Duplicated(tetr_dat, d_tetr_dat), Duplicated(out_sampled_points, d_out_sampled_points), Duplicated(source_arr, d_source_arr), Duplicated(control_points, d_control_points), Duplicated(sv_centers, d_sv_centers), Const(num_base_samp_points), Const(num_additional_samp_points))
    
            return nothing
    end
    
    
    function call_point_info_kern_add(tetr_dat, out_sampled_points, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads, blocks, pad_point_info)

        #shmem is in bytes
        tetr_shape = size(tetr_dat)
        out_shape = size(out_sampled_points)
        to_pad_tetr = CUDA.ones(pad_point_info, tetr_shape[2], tetr_shape[3]) * 2
        tetr_dat = vcat(tetrs, to_pad_tetr)
    
        to_pad_out = CUDA.ones(pad_point_info, out_shape[2], out_shape[3]) * 2
        out_sampled_points = vcat(out_sampled_points, to_pad_out)
    
    
        #@device_code_warntype  @cuda threads = threads blocks = blocks point_info_kern(CuStaticSharedArray(Float32, (128,3)),tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points)
        @cuda threads = threads blocks = blocks point_info_kern_add_a(tetr_dat, out_sampled_points, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points)
    
    
        tetr_dat = tetr_dat[1:tetr_shape[1], :, :]
        out_sampled_points = out_sampled_points[1:out_shape[1], :, :]
    
        # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
        return out_sampled_points
    end
    
    
    
    # rrule for ChainRules.
    function ChainRulesCore.rrule(::typeof(call_point_info_kern_add), tetr_dat, out_sampled_points, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, pad_point_info)
    
    
        out_sampled_points = call_point_info_kern_add(tetr_dat, out_sampled_points, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, pad_point_info)    #TODO unhash
        d_tetr_dat = CUDA.ones(size(tetr_dat)...)
        # d_out_sampled_points = CUDA.ones(size(out_sampled_points)...) # TODO remove
        d_source_arr = CUDA.ones(size(source_arr)...)
        d_control_points = CUDA.ones(size(control_points)...)
        d_sv_centers = CUDA.ones(size(sv_centers)...)
    
    
        #pad to avoid conditionals in kernel
        tetr_shape = size(tetr_dat)
        out_shape = size(out_sampled_points)
        to_pad_tetr = CUDA.ones(pad_point_info, tetr_shape[2], tetr_shape[3])
        tetr_dat = vcat(tetrs, to_pad_tetr)
        d_tetr_dat = vcat(d_tetr_dat, to_pad_tetr)
    
        to_pad_out = CUDA.ones(pad_point_info, out_shape[2], out_shape[3])
        out_sampled_points = vcat(out_sampled_points, to_pad_out)
    
    
        function call_test_kernel1_pullback(d_out_sampled_points)   
    
            d_out_sampled_points = vcat(CuArray(collect(d_out_sampled_points)), to_pad_out)
    
            @device_code_warntype @cuda threads = threads_point_info blocks = blocks_point_info point_info_kern_deff(tetr_dat, d_tetr_dat, out_sampled_points, d_out_sampled_points, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, num_base_samp_points, num_additional_samp_points
            ) 
    
    
            d_tetr_dat = d_tetr_dat[1:tetr_shape[1], :, :]
            d_out_sampled_points = d_out_sampled_points[1:out_shape[1], :, :]
    
            return NoTangent(), d_tetr_dat, d_out_sampled_points, d_source_arr, d_control_points, d_sv_centers, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
    

        return out_sampled_points, call_test_kernel1_pullback
    
    end
    

    function prepare_for_point_info_kern(tetr_dat_shape)
        threads = 128
        needed_blocks = ceil(Int, tetr_dat_shape[1] / threads)
        to_pad = (threads * needed_blocks) - tetr_dat_shape[1]
    
        return threads, needed_blocks, to_pad
    end


    function call_point_info_kern_test(tetr_dat, source_arr, control_points, threads, blocks, pad_point_info, num_base_samp_points, num_additional_samp_points)

        tetr_shape = size(tetr_dat)
        to_pad_tetr = ones(pad_point_info, tetr_shape[2], tetr_shape[3]) * 2
        tetr_dat = vcat(tetr_dat, to_pad_tetr)
    
        tetr_dat = CuArray(Float32.(tetr_dat))
        source_arr = CuArray(Float32.(source_arr))
        control_points = CuArray(Float32.(control_points))
        out_sampled_points = CUDA.zeros((size(tetr_dat)[1], num_base_samp_points + (3 * num_additional_samp_points), 5))
        out_shape = size(out_sampled_points)
        to_pad_out = CUDA.ones(pad_point_info, out_shape[2], out_shape[3]) * 2
        out_sampled_points = vcat(out_sampled_points, to_pad_out)
        # @cuda threads = threads blocks = blocks point_info_kern(CuStaticSharedArray(Float32, (128,3)),tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points)
        @cuda threads = threads blocks = blocks point_info_kern_forward(tetr_dat, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points)
        out_sampled_points = out_sampled_points[1:out_shape[1], :, :]
    
    
    
    
    
        # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
        return out_sampled_points
    end

    
    radiuss = Float32(4.0)
    diam = radiuss * 2
    num_weights_per_point = 6
    a = 36
    image_shape = (a, a, a)

    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
    sv_centers, control_points, tetrs, dims = example_set_of_svs
    #here we get all tetrahedrons mapped to non modified locations
    sv_tetrs = map(index -> fill_tetrahedron_data(tetrs, sv_centers, control_points, index), 1:(size(tetrs)[1]))
    source_arr = rand(Float32, image_shape)
    num_base_samp_points, num_additional_samp_points = 3, 2

    tetr_dat_out = zeros(size(tetrs))

    threads_point_info, blocks_point_info, pad_point_info = prepare_for_point_info_kern(size(tetrs))
    out_sampled_points = call_point_info_kern_test(tetrs, source_arr, control_points, threads_point_info, blocks_point_info, pad_point_info, num_base_samp_points, num_additional_samp_points)




Enzyme.autodiff_deferred(Enzyme.Reverse, point_info_kern_forward, Const, Duplicated(tetr_dat, d_tetr_dat), Duplicated(out_sampled_points, d_out_sampled_points), Duplicated(source_arr, d_source_arr), Duplicated(control_points, d_control_points), Duplicated(sv_centers, d_sv_centers), Const(num_base_samp_points), Const(num_additional_samp_points))
