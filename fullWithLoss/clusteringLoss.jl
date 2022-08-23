using Revise
using Statistics
"""
my mean
"""
function my3dMean(arr,Nx,Ny,Nz)
    sum=0.0
    for x in 1:Nx, y in 1:Ny, z in 1:Nz
        sum+=arr[x,y,z]
    end#for    
    return sum/(Nx*Ny*Nz)
end    

function variance3d( arr,Nx,Ny,Nz )
    mean=my3dMean(arr,Nx,Ny,Nz)
    varr=0
    for x in 1:Nx, y in 1:Ny, z in 1:Nz
        varr+= ( (mean-arr[x,y,z])^2)
    end#for  
    return varr/(Nx*Ny*Nz)
end#variance

function my1dsum(arr,Nx)
    sum=0
    for x in 1:Nx
        sum+=arr[x]
    end#for    
    return sum
end    


function my1dMean(arr,Nx)
    return my1dsum(arr,Nx) /(Nx)
end    

function variance1d( arr,Nx)
    mean=my1dMean(arr,Nx)
    varr=0
    for x in 1:Nx
        varr+= ( (mean-arr[x])^2)
    end#for  
    return varr/(Nx)
end#variance



"""
test loss assuming that in each corner of the daa cube there should 
be region with diffrent mean, small variance and big diffrence relative to other regions 
for simplicity curently it will run on CPU
"""
function clusteringLossTest(outLoss,Nx, Ny, Nz,crossBorderWhere, oneSidePad,A,tops,tope, bottoms, bottome, lefts, lefte, rights, righte, anteriors,anteriore ,posteriors , posteriore)
    outLoss[1]=15.0

    top_left_post =A[tops:tope,lefts:lefte, posteriors:posteriore ] 
    top_right_post =A[tops:tope,rights:righte, posteriors:posteriore ] 

    top_left_ant =A[tops:tope,lefts:lefte, anteriors:anteriore ] 
    top_right_ant =A[tops:tope,rights:righte, anteriors:anteriore ] 

    bottom_left_post =A[bottoms:bottome,lefts:lefte, posteriors:posteriore ] 
    bottom_right_post =A[bottoms:bottome,rights:righte, posteriors:posteriore ] 

    bottom_left_ant =A[bottoms:bottome,lefts:lefte, anteriors:anteriore ] 
    bottom_right_ant =A[bottoms:bottome,rights:righte, anteriors:anteriore ] 

    means=(
        mean(top_left_post),mean(top_right_post)
        ,mean(top_left_ant),mean(top_right_ant)
        ,mean(bottom_left_post),mean(bottom_right_post)
        ,mean(bottom_right_ant),mean(bottom_right_ant)    
    )
    variances= var(top_left_post)+var(top_right_post)
    +var(top_left_ant)+var(top_right_ant)
    +var(bottom_left_post)+var(bottom_right_post)
    +var(bottom_right_ant)+var(bottom_right_ant)    

    #print( " aaa $(variances)"  )

    #regions= (top_left_post,top_right_post,top_left_ant,top_right_ant,bottom_left_post, bottom_right_post ,bottom_left_ant,bottom_right_ant)
    #my3dMean(top_left_post,crossBorderWhere,crossBorderWhere,crossBorderWhere)

    # sum=0.0
    # for x in 1:5
    #     sum+=top_left_post[x]
    # end#for    
    # return sum/5
    
#     #means=map( arr-> my3dMean(arr,crossBorderWhere,crossBorderWhere,crossBorderWhere) ,regions)
#     # variances=map(arr-> variance3d(arr,crossBorderWhere,crossBorderWhere,crossBorderWhere) ,regions)
    
    variance_ofMeans=var(means)

#     # #variance in region should be small but big between regions
    # outLoss[1]= (sum(variances) -(variance_ofMeans*6))
    # outLoss[1]= (sum(variances) )#-variance_ofMeans
    outLoss[1]= variances#-variance_ofMeans
    #outLoss[1]= 15.0#-variance_ofMeans
    
    return 11.0
end

function clusteringLossTestDeff(outLoss,doutLoss,Nx, Ny, Nz,crossBorderWhere, oneSidePad, A, dA,tops,tope
    , bottoms, bottome, lefts, lefte, rights, righte, anteriors,anteriore ,posteriors , posteriore)
    return Enzyme.autodiff(Reverse, clusteringLossTest, Active,Duplicated(outLoss,doutLoss) #this tell about return
    , Const(Nx), Const(Ny)
    , Const(Nz), Duplicated(A, dA) , Const(crossBorderWhere), Const(oneSidePad)
    ,Const(tops),Const(tope), Const(bottoms), Const(bottome), Const(lefts), Const(lefte), Const(rights)
    , Const(righte), Const(anteriors) , Const(anteriore),Const(posteriors) , Const(posteriore))
end
# clusteringLossTest(Nx, Ny, Nz,crossBorderWhere, oneSidePad,A, p)