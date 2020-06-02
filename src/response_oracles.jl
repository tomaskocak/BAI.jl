# include("general.jl")
# include("reducers.jl")

function response_vanilla(mu::Array{<:Real}, omega::Array{<:Real}, lambda::Array{<:Real}, i::Int64, j::Int64, epsilon::Float64)::Array{Float64}
    lambda_response = copy(lambda)
    lambda_response[i] = (mu[i] * omega[i] + mu[j] * omega[j] - epsilon * omega[j]) / (omega[i] + omega[j])
    lambda_response[j] = lambda_response[i] + epsilon
    return copy(lambda_response)
end

function response_spectral(mu::Array{<:Real}, omega::Array{<:Real}, lambda::Array{<:Real}, i::Int64, j::Int64, epsilon::Float64, laplacian::Array{<:Real}, R::Real)::Array{Float64}
    lambda_vanilla = response_vanilla(mu, omega, lambda, i, j, epsilon)
end

function response_oracle()
    # call either reponse_vanilla or response_spectral
end

