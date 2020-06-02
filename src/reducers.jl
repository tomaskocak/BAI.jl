function reduce_index(a, j)
    return a < j ? a : a - 1
end

function recover_index(a, j)
    return a < j ? a : a + 1
end

function reduce_omega(omega::Array{<:Real}, i::Int64, j::Int64)::Array{Float64}
    omega_tilde = copy(omega)
    omega_tilde[i] += omega_tilde[j]
    omega_tilde = omega_tilde[[1:(j - 1); (j + 1):end]]
    return copy(omega_tilde)
end

function reduce_lambda(lambda::Array{<:Real}, j::Int64)::Array{Float64}
    new_lambda = copy(lambda)[[1:(j - 1); (j + 1):end]]
    return copy(new_lambda)
end

function recover_lambda(lambda_tilde::Array{<:Real}, i::Int64, j::Int64, epsilon::Real)::Array{Float64}
    lambda = [lambda_tilde[1:(j - 1)]; lambda_tilde[reduce_index(i, j)] + epsilon ; lambda_tilde[j:end]]
    return copy(lambda)
end

function reduce_mu(mu::Array{<:Real}, omega::Array{<:Real}, i::Int64, j::Int64, epsilon::Real)::Array{Float64}
    mu_tilde = mu[[1:(j - 1); (j + 1):end]]
    mu_tilde[reduce_index(i, j)] = (omega[i] * mu[i] + omega[j] * mu[j]) / (omega[i] + omega[j]) - (epsilon * omega[i]) / (omega[i] + omega[j])
    return copy(mu_tilde)
end

function reduce_laplacian(Laplacian::Array{<:Real}, i::Int64, j::Int64)::Array{Float64}
    Laplacian_tilde = copy(Laplacian)
    Laplacian_tilde[i,:] += Laplacian_tilde[j,:]
    Laplacian_tilde[:,i] += Laplacian_tilde[:,j]
    Laplacian_tilde = Laplacian_tilde[[1:(j - 1); (j + 1):end], [1:(j - 1); (j + 1):end]]
    # Laplacian_tilde[reduce_index(i, j), reduce_index(i, j)] -= sum(Laplacian_tilde[reduce_index(i, j), :])
    return copy(Laplacian_tilde)
end

function reduce_laplacian_j(Laplacian::Array{<:Real}, i::Int64, j::Int64)::Array{Float64}
    Laplacian_tilde_j = Laplacian[:, j]
    Laplacian_tilde_j[j] += Laplacian_tilde_j[i]
    Laplacian_tilde_j = Laplacian_tilde_j[[(1:i - 1); (i + 1):end]]
    return copy(Laplacian_tilde_j)
end
