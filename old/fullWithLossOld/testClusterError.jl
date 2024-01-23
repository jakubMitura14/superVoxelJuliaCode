using Lux, NNlib, Optimisers, Plots, Random, Statistics, Zygote,CUDA
#Lux v0.4.14

#variables
dim_x,dim_y,dim_z = 34, 34, 34
rng = Random.default_rng()
#setting up convolutions
conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1,pad=Lux.SamePad())
function getConvModel()
    return Lux.Chain(conv1(1,4),conv1(4,8),conv1(8,4),conv1(4,2),conv1(2,1))
end#getConvModel
#defining model, states, parameters,Optimisers
model = getConvModel()
ps, st = Lux.setup(rng, model)
opt = Optimisers.Adam()
#loss
function loss_function(model, ps, st, x)
    y_pred, st = Lux.apply(model, x, ps, st)
    return -1*(sum(y_pred)), st, ()
end
#Lux objects
tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
vjp_rule = Lux.Training.ZygoteVJP()
#main iteration loop
function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data,
    epochs::Int)
   # data = data .|> Lux.gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
                                                                data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end
# dummy data
x = randn(rng, Float32, dim_x,dim_y,dim_z)
x =reshape(x, (dim_x,dim_y,dim_z,1,1))
#execute
#works
y_pred, st =Lux.apply(model, x, ps, st) 
size(y_pred)
#breaks during backpropagation
tstate = main(tstate, vjp_rule, CuArray(x),1)
