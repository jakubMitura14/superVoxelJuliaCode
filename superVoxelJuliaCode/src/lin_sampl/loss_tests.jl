using Revise
using Meshes
using LinearAlgebra
using GLMakie
using Combinatorics
using SplitApplyCombine
using CUDA
using Combinatorics
using Random
using Statistics
using ChainRulesCore
using Test
using ChainRulesTestUtils
using EnzymeTestUtils
using Logging,FiniteDifferences,FiniteDiff
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_tetr.jl")
includet("/home/jm/projects_new/superVoxelJuliaCode/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")


radiuss = Float32(4.0)
diam = radiuss * 2
num_weights_per_point = 6
a = 36
image_shape = (a, a, a)

example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs

dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
# control_points first dimension is lin_x, lin_y, lin_z, oblique
weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
# # weights = rand(image_shape...)
# weights = weights .- 0.50001
# weights = (weights) .* 100
# weights = Float32.(tanh.(weights * 0.02))



weights = ones(size(weights)...) .* 0.5

control_points_non_modified = copy(control_points)

control_points_size = size(control_points)

threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(control_points_size, size(weights))

# control_points = call_apply_weights_to_locs_kern(CuArray(control_points), CUDA.zeros(size(control_points)...), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w)
control_points=call_apply_weights_to_locs_kern(CuArray(control_points),CuArray(copy(control_points)),CuArray(weights),radiuss,threads_apply_w,blocks_apply_w)

# """
# this function test single point in the control_points and check weather change in value is correct generally if weight is 
# changed the control point should change as well if it is for example positive 0.5 then the control point should be moved in the direction of the axis of change
#     0.5*radius a
# """
# @testset "Control Point Tests" begin

#     function test_control_point(cart_index,axes_of_change,point_ind,weight,radius,control_points,control_points_non_modified)
#         old_p=control_points_non_modified[cart_index[1],cart_index[2],cart_index[3],point_ind,:,:]
#         new_p=control_points[cart_index[1],cart_index[2],cart_index[3],point_ind,:,:]
#         changes=[0.0,0.0,0.0]
#         for ax in axes_of_change
#             changes[ax]=weight*radius
#         end
#         old_p=old_p.+changes
#         return @test old_p==new_p
#     end

#     for cart in [(1,1,1),(2,2,2)]
#         test_control_point((1,1,1),(1),1,0.5,radiuss,Array(control_points),Array(control_points_non_modified))
#         test_control_point((1,1,1),(2),2,0.5,radiuss,Array(control_points),Array(control_points_non_modified))
#         test_control_point((1,1,1),(3),3,0.5,radiuss,Array(control_points),Array(control_points_non_modified))
#         test_control_point((1,1,1),(1,2,3),4,0.5,radiuss,Array(control_points),Array(control_points_non_modified))
#     end
# end

# test_reverse(apply_weights_to_locs_kern,Active,control_points,control_points_non_modified,weights,radiuss,UInt32(size(control_points)[1]), UInt32(size(control_points)[2]), UInt32(size(control_points)[3]) )

# test_rrule(call_apply_weights_to_locs_kern, CuArray(control_points),CuArray(copy(control_points)),CuArray(weights), radiuss, threads_apply_w, blocks_apply_w);


weights=CuArray(Float32.(weights))
control_points=CuArray(Float32.(control_points))
control_points_out = CuArray(copy(Float32.(control_points)))


# test_reverse(apply_weights_to_locs_kern,Active,control_points,control_points_out,weights,radiuss,UInt32(size(control_points)[1]), UInt32(size(control_points)[2]), UInt32(size(control_points)[3]) )


# cp_x,cp_y,cp_z=UInt32(size(control_points)[1]), UInt32(size(control_points)[2]), UInt32(size(control_points)[3]) 

##################################



"""
adapted from https://github.com/JuliaDiff/ChainRulesTestUtils.jl/blob/ea867a50896d3a9d31f311b5f7fb098eca1620fe/src/finite_difference_calls.jl#L19
    _wrap_function(f, xs, ignores)

Return a new version of `f`, `fnew`, that ignores some of the arguments `xs`.

# Arguments
- `f`: The function to be wrapped.
- `xs`: Inputs to `f`, such that `y = f(xs...)`.
- `ignores`: Collection of `Bool`s, the same length as `xs`.
  If `ignores[i] === true`, then `xs[i]` is ignored; `∂xs[i] === NoTangent()`.
"""
function _wrap_function(f, xs, ignores)
    function fnew(sigargs...)
        callargs = Any[]
        j = 1

        for (i, (x, ignore)) in enumerate(zip(xs, ignores))
            if ignore
                push!(callargs, x)
            else
                push!(callargs, sigargs[j])
                j += 1
            end
        end
        @assert j == length(sigargs) + 1
        @assert length(callargs) == length(xs)
        return f(callargs...)
    end
    return fnew
end

"""
adapted from https://github.com/JuliaDiff/ChainRulesTestUtils.jl/blob/ea867a50896d3a9d31f311b5f7fb098eca1620fe/src/finite_difference_calls.jl#L28

    _make_j′vp_call(fdm, f, ȳ, xs, ignores) -> Tuple

Call `FiniteDifferences.j′vp`, with the option to ignore certain `xs`.

# Arguments
- `fdm::FiniteDifferenceMethod`: How to numerically differentiate `f`.
- `f`: The function to differentiate.
- `ȳ`: The adjoint w.r.t. output of `f`.
- `xs`: Inputs to `f`, such that `y = f(xs...)`.
- `ignores`: Collection of `Bool`s, the same length as `xs`.
  If `ignores[i] === true`, then `xs[i]` is ignored; `∂xs[i] === NoTangent()`.

# Returns
- `∂xs::Tuple`: Derivatives estimated by finite differencing.
"""
function _make_j′vp_call(fdm, f, ȳ, xs, ignores)
    f2 = _wrap_function(f, xs, ignores)

    ignores = collect(ignores)
    args = Any[NoTangent() for _ in 1:length(xs)]
    all(ignores) && return (args...,)
    sigargs = xs[.!ignores]
    arginds = (1:length(xs))[.!ignores]
    fd = j′vp(fdm, f2, ȳ, sigargs...)
    @assert length(fd) == length(arginds)

    for (dx, ind) in zip(fd, arginds)
        args[ind] = ProjectTo(xs[ind])(dx)
    end
    return (args...,)
end


function to_vec(m::CuDeviceVector)
    return m

end


function to_vec(x::Tuple)
    return x
    # x_vecs_and_backs = map(to_vec, x)
    # x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    # lengths = map(length, x_vecs)
    # sz = typeof(lengths)(cumsum(collect(lengths)))
    # function Tuple_from_vec(v)
    #     map(x_backs, lengths, sz) do x_back, l, s
    #         return x_back(v[s - l + 1:s])
    #     end
    # end
    # return reduce(vcat, x_vecs), Tuple_from_vec
end

struct PrimalAndTangent{P,D}
    primal::P
    tangent::D
end
f=call_apply_weights_to_locs_kern
args=(control_points,control_points_out,weights, radiuss, threads_apply_w, blocks_apply_w)
primal(p) = p.primal
tangent(p) = p.tangent
auto_primal_and_tangent(primal; rng=Random.GLOBAL_RNG) = primal ⊢ rand_tangent(rng, primal)
auto_primal_and_tangent(both::PrimalAndTangent; kwargs...) = both
primals_and_tangents = auto_primal_and_tangent.((f, args...))
primals = primal.(primals_and_tangents)
accum_cotangents = tangent.(primals_and_tangents)
is_ignored = isa.(accum_cotangents, NoTangent)


y=f(args...)
output_tangent=Auto()
ȳ = output_tangent isa Auto ? rand_tangent(y) : output_tangent

# FiniteDiff.finite_difference_jacobian(f,(control_points,control_points_out,weights, radiuss, threads_apply_w, blocks_apply_w))
fdm=central_fdm(5, 1)
fd_cotangents = _make_j′vp_call(fdm, f, ȳ, primals, is_ignored)




# fdm=central_fdm(5, 1)
# j′vp(central_fdm(5, 1), f, control_points,control_points_out,weights, radiuss, threads_apply_w, blocks_apply_w)



struct PrimalAndTangent{P,D}
    primal::P
    tangent::D
end
primal(p) = p.primal
# primal(p::ChainRulesTestUtils.PrimalAndTangent{CuArray{Float32, 5, CUDA.DeviceMemory}, CuArray{Float32, 5, CUDA.DeviceMemory}}) = p.primal
# primal(p::ChainRulesTestUtils.PrimalAndTangent{CuArray{Float32, 4, CUDA.DeviceMemory}, CuArray{Float32, 4, CUDA.DeviceMemory}}) = p.primal
tangent(p) = p.tangent
auto_primal_and_tangent(primal; rng=Random.GLOBAL_RNG) = primal ⊢ rand_tangent(rng, primal)
auto_primal_and_tangent(both::PrimalAndTangent; kwargs...) = both

_string_typeof(x) = string(typeof(x))
_string_typeof(xs::Tuple) = join(_string_typeof.(xs), ",")
_string_typeof(x::PrimalAndTangent) = _string_typeof(primal(x))  # 
TEST_INFERRED=false
"""
    @maybe_inferred [Type] f(...)

Like `@inferred`, but does not check the return type if tests are run as part of PkgEval or
if the environment variable `CHAINRULES_TEST_INFERRED` is set to `false`.
"""
macro maybe_inferred(ex...)
    inferred = Expr(:macrocall, GlobalRef(Test, Symbol("@inferred")), __source__, ex...)
    return :(TEST_INFERRED[] ? $(esc(inferred)) : $(esc(last(ex))))
end

"""
    _test_inferred(f, args...; kwargs...)

Simple wrapper for [`@maybe_inferred f(args...: kwargs...)`](@ref `@maybe_inferred`), avoiding the type-instability in not
knowing how many `kwargs` there are.
"""
function _test_inferred(f, args...; kwargs...)
    if isempty(kwargs)
        @maybe_inferred f(args...)
    else
        @maybe_inferred f(args...; kwargs...)
    end
end



function _is_inferrable(f, args...; kwargs...)
    try
        _test_inferred(f, args...; kwargs...)
        return true
    catch ErrorException
        return false
    end
end



function test_rrule(
    config::typeof(call_apply_weights_to_locs_kern),
    f,
    args...;
    output_tangent=Auto(),
    check_thunked_output_tangent=true,
    fdm=central_fdm(5, 1),
    rrule_f=ChainRulesCore.rrule,
    check_inferred::Bool=true,
    fkwargs::NamedTuple=NamedTuple(),
    rtol::Real=1e-9,
    atol::Real=1e-9,
    testset_name=nothing,
    kwargs...,
)
    # To simplify some of the calls we make later lets group the kwargs for reuse
    isapprox_kwargs = (; rtol=rtol, atol=atol, kwargs...)
    testset_name = testset_name === nothing ? "test_rrule: $f on $(_string_typeof(args))" : testset_name
    # and define helper closure over fkwargs
    call(f, xs...) = f(xs...; fkwargs...)
    # config=convert(RuleConfig,config)
    # Base.convert(::Type{MyType}, x::Any)
    # config = RuleConfig(config)
    
    # @testset "$(testset_name)" begin

        # Check correctness of evaluation.
        # print("ffffffffffffffffffffffffff \n $(map( typeof, args)) \n")
        primals_and_tangents = auto_primal_and_tangent.((f, args...))
        primals = primal.(primals_and_tangents)
        accum_cotangents = tangent.(primals_and_tangents)

        # if check_inferred && _is_inferrable(primals...; fkwargs...)
            # _test_inferred(rrule_f, config, primals...; fkwargs...)
        # end
        res = rrule_f(config, primals...; fkwargs...)
        res === nothing && throw(MethodError(rrule_f, Tuple{Core.Typeof.(primals)...}))
        y_ad, pullback = res
        # y = call(primals...)
        y = call(f, xs...) = f(xs...; fkwargs...)
        test_approx(y_ad, y, "Failed primal value check"; isapprox_kwargs...)  # make sure primal is correct

        ȳ = output_tangent isa Auto ? rand_tangent(y) : output_tangent

        check_inferred && _test_inferred(pullback, ȳ)
        ad_cotangents = pullback(ȳ)
        @info(
            "The pullback must return a Tuple (∂self, ∂args...)",
            ad_cotangents isa Tuple
        )
        @info(
            "The pullback should return 1 cotangent for the primal and each primal input.",
            length(ad_cotangents) == length(primals)
        )

        # Correctness testing via finite differencing.
        is_ignored = isa.(accum_cotangents, NoTangent)
        fd_cotangents = _make_j′vp_call(fdm, call, ȳ, primals, is_ignored)
        msgs = ntuple(i->"cotangent for input $i, $(summary(fd_cotangents[i]))", length(fd_cotangents))
        foreach(accum_cotangents, ad_cotangents, fd_cotangents, msgs) do args...
            _test_cotangent(args...; check_inferred=check_inferred, isapprox_kwargs...)
        end

        if check_thunked_output_tangent
            test_approx(ad_cotangents, pullback(@thunk(ȳ)), "pulling back a thunk:"; isapprox_kwargs...)
            check_inferred && _test_inferred(pullback, @thunk(ȳ))
        end
    # end  # top-level testset
end



test_rrule(call_apply_weights_to_locs_kern, control_points,control_points_out,weights, radiuss, threads_apply_w, blocks_apply_w)


# @cuda threads = (1) blocks = (1) apply_weights_to_locs_kern_deff(UInt32(sizz[3]))
# @cuda threads = (1) blocks = 1(1) apply_weights_to_locs_kern_deff(control_points,d_control_points,Float32(4.0), UInt32(sizz[1]), UInt32(sizz[2]), UInt32(sizz[3]))
# @cuda threads = threads_apply_w blocks = blocks_apply_w  Enzyme.autodiff_deferred(Enzyme.Reverse, apply_weights_to_locs_kern, Const
# , Duplicated(control_points, d_control_points)
# , Duplicated(control_points_out, d_control_points_out)
# , Duplicated(weights, d_weights)
# , Const(radius)
# , Const(cp_x)
# , Const(cp_y)
# , Const(cp_z)
# )
# @cuda threads = threads_apply_w blocks = blocks_apply_w apply_weights_to_locs_kern_deff(
#             control_points,d_control_points  
#             , control_points_out, d_control_points_out
#             , weights, d_weights
#             , Float32(4.0), UInt32(sizz[1]), UInt32(sizz[2]), UInt32(sizz[3]))


