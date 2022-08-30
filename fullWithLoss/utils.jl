"""pdf of the univariate normal distribution.
    μ-mean
    σ - variance
    TODO( change into log probability)
"""
@inline function univariate_normal(x, μ, σ)
    return ((1.0/sqrt(2 * π * σ)) *exp(-(x - μ)^2 / (2 * σ)))
end#univariate_normal

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
    return ((normPaira(a, b) / (normPaira(a, b) + normPairb(a, b))) * a) + ((normPairb(a, b) / (normPaira(a, b) + normPairb(a, b))) * b)
end#alaMax