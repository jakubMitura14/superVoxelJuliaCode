
using Flux,Lux, Random,Optimisers


"""
get transposed convolution from flux to Lux
# some example for convolution https://github.com/avik-pal/Lux.jl/blob/main/lib/Boltz/src/vision/vgg.jl
# 3D layer utilities from https://github.com/Dale-Black/MedicalModels.jl/blob/master/src/utils.jl

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


conv1 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=1, pad=Lux.SamePad())
conv2 = (in, out) -> Lux.Conv((3,3,3),  in => out , NNlib.tanh, stride=2, pad=Lux.SamePad())
tran = ( in, out) -> Flux.ConvTranspose((3, 3, 3), in=>out, stride=2, pad=Flux.SamePad())

tran2(in_chan,out_chan) = FluxCompatLayer(tran(in_chan,out_chan))

dim_x,dim_y,dim_z=32,32,32
base_arr=rand(dim_x,dim_y,dim_z )
# vggModel=vgg_block(1,1,2)
rng = Random.MersenneTwister()
base_arr=reshape(base_arr, (dim_x,dim_y,dim_z,1,1))
#model = Lux.Chain(Lux.Dense(100, 13200, NNlib.tanh),  )

in_chs=1
lbl_chs=1
# Contracting layers
# l1 = Chain(conv1(in_chs, 4))
# l2 = Chain(l1, conv1(4, 4), conv2(4, 16))
# l3 = Chain(l2, conv1(16, 16), conv2(16, 32))
# l4 = Chain(l3, conv1(32, 32), conv2(32, 64))
# l5 = Chain(l4, conv1(64, 64), conv2(64, 128))

# # Expanding layers
# l6 = Chain(l5, tran2(128, 64), conv1(64, 64))
# l7 = Chain(Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))       # Residual connection between l6 & l4
# l8 = Chain(Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))       # Residual connection between l7 & l3
# l9 = Chain(Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))          # Residual connection between l8 & l2
# l10 = Chain(l9, conv1(4, lbl_chs))



l1 = conv2(1, 1)
l2 = conv2(1, 1)
ll1=Lux.Chain(l1,l2)

l3=tran2(1, 1)
l4=tran2(1, 1)
l5=tran2(1, 1)

model = Lux.Parallel(+, Lux.Chain(ll1,l3), l2)

rng = Random.MersenneTwister()
base_arr=reshape(base_arr, (dim_x,dim_y,dim_z,1,1))

ps, st = Lux.setup(rng, model)
out = Lux.apply(model, base_arr, ps, st)

ps, st = Lux.setup(rng, Lux.Chain(l1,l2))
out = Lux.apply(Lux.Chain(l1,l2), base_arr, ps, st)

size(out[1]) # we gat smaller 

mm=Lux.Chain(l1,l2)


model = Lux.Chain(
    l1, ##63,63,61
    #,Lux.Parallel(+, l1, l2)
    # tran2(lbl_chs, lbl_chs)
    # ,Lux.Conv((3,3,3),  1 => 1 , NNlib.tanh, stride=1,pad=2)
     )
model = Lux.Parallel(+, conv2(1, 1), tran2(1, 1))

#model = Lux.Chain(l1, tran2(4, lbl_chs))


rng = Random.MersenneTwister()
base_arr=reshape(base_arr, (dim_x,dim_y,dim_z,1,1))

ps, st = Lux.setup(rng, model)
out = Lux.apply(model, base_arr, ps, st)
size(out[1]) # we gat smaller 