using StatsFuns
"""pdf of the univariate normal distribution.
    μ-mean
    σ - variance
    TODO( change into log probability)
"""
#from https://github.com/JuliaStats/StatsFuns.jl/blob/13e231a0a22e716426b73cb87ff3b8b24e33aaf1/src/distrs/norm.jl
zval(μ::Real, σ::Real, x::Number) = (x - μ) / σ
@inline  univariate_normal(μ::Real, σ::Real, x::Number) = exp(-abs2( zval(μ, one(σ), x)   )/2) * invsqrt2π

# @inline function univariate_normal(z, μ, σ)
#     lognorm.pdf(x,s) = 1/(s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)

#     return exp(-abs2(z)/2) * invsqrt2π

# end#univariate_normal

"""
for some reason raising to the power greater then 2 give erro - hence this macro enable caling power of two multiple times
"""
macro myPowTwo(ex, num)
    exIn = quote
        (($ex)^2)
    end
    for i in 1:(num-1)
        exIn = quote
            (($exIn)^2)
        end
    end
    return esc(:($exIn))
end



@inline function normPaira(a::Float32, b::Float32)::Float32
    return @myPowTwo((a / (a + b)) + 1, 5)
end#normPair

@inline function normPairb(a::Float32, b::Float32)::Float32
    return @myPowTwo((b / (a + b)) + 1, 5)
end#normPair

"""
given 2 numbers return sth like max
"""
@inline function alaMax(a, b)::Float32
    return ((@myPowTwo((a / (a + b)) + 1, 5) / (@myPowTwo((a / (a + b)) + 1, 5) + @myPowTwo((b / (a + b)) + 1, 5))) * a)  + ((@myPowTwo((b / (a + b)) + 1, 5) / (@myPowTwo((a / (a + b)) + 1, 5) + @myPowTwo((b / (a + b)) + 1, 5))) * b)
end#alaMax


