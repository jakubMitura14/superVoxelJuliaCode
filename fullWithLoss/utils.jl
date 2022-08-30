"""pdf of the univariate normal distribution.
    μ-mean
    σ - variance
"""
function univariate_normal(x, μ, σ)
    return ((1.0/sqrt(2 * π * σ)) *exp(-(x - μ)^2 / (2 * σ)))
end#univariate_normal