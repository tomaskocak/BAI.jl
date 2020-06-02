function smoothness(lambda::Array{<:Real}, laplacian::Array{<:Real})::Float64
    return lambda' * laplacian * lambda
end

function objective(mu::Array{<:Real}, omega::Array{<:Real}, lambda::Array{<:Real})::Float64
    return omega' * (mu - lambda).^2 / 2
end
