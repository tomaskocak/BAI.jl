module BAI

using LinearAlgebra
# using JLD
using Plots

export Setting

##################################################

mutable struct Setting
    K::Int64
    mu::Vector{Float64}
    epsilon::Float64
    laplacian::Array{Float64,2}
    R::Float64
    omega::Vector{Float64}
end

##################################################

function create_setting(mu::Array{<:Real})
    return Setting(length(mu), mu, 0, zeros(length(mu), length(mu)), 0, ones(length(mu))./length(mu))
end

##################################################

function create_setting(mu::Array{<:Real}, epsilon::Real)
    return Setting(length(mu), mu, epsilon, zeros(length(mu), length(mu)), 0, ones(length(mu))./length(mu))
end

##################################################

function create_setting(mu::Array{<:Real}, epsilon::Real, laplacian::Array{<:Real}, R::Real)
    return Setting(length(mu), mu, epsilon, laplacian, R, ones(length(mu))./length(mu))
end

##################################################

function best_arms(setting::Setting)::Array{Int64}
    return [i for i in 1:setting.K if setting.mu[i] >= maximum(setting.mu) - setting.epsilon - 1e-13]
end

##################################################

function smoothness(lambda::Array{<:Real}, laplacian::Array{<:Real})::Float64
    # println("----------")
    # println(size(lambda))
    # println(size(laplacian))
    # println(lambda' * laplacian * lambda)
    return lambda' * laplacian * lambda
end

##################################################

function objective(mu::Array{<:Real}, omega::Array{<:Real}, lambda::Array{<:Real})::Float64
    return omega' * (mu - lambda).^2 / 2
end

##################################################

function objective(setting::Setting, lambda::Array{<:Real})::Float64
    return setting.omega' * (setting.mu - lambda).^2 / 2
end

##################################################

function reduce_index(a, j)
    return a < j ? a : a - 1
end

##################################################

function recover_index(a, j)
    return a < j ? a : a + 1
end

##################################################

function reduce_omega(omega::Array{<:Real}, i::Int64, j::Int64)::Array{Float64}
    omega_tilde = copy(omega)
    omega_tilde[i] += omega_tilde[j]
    omega_tilde = omega_tilde[[1:(j - 1); (j + 1):end]]
    return copy(omega_tilde)
end

##################################################

function reduce_lambda(lambda::Array{<:Real}, j::Int64)::Array{Float64}
    # Function that produces K-1 dimensional vector lambda_tilde from vector lambda 
    # by removing ind_rem element of lambda. This is thanks to the fact that ind_rem
    # and ind_keep are supposed to be equal in the case of the closes bandit problem
    # from Alt_{ind_keep}(mu)

    new_lambda = copy(lambda)[[1:(j - 1); (j + 1):end]]
    return copy(new_lambda)
end

##################################################

function recover_lambda(lambda_tilde::Array{<:Real}, i::Int64, j::Int64, epsilon::Real)::Array{Float64}
    # Recovering vector lambda from lambda_tilde. Lambda_tilde was created by removing ind_rem
    # element while ind_rem and ind_keep were supposed to be the same.

    lambda = [lambda_tilde[1:(j - 1)]; lambda_tilde[reduce_index(i, j)] + epsilon ; lambda_tilde[j:end]]
    return copy(lambda)
end

##################################################

function reduce_mu(mu::Array{<:Real}, omega::Array{<:Real}, i::Int64, j::Int64, epsilon::Real)::Array{Float64}
    mu_tilde = mu[[1:(j - 1); (j + 1):end]]
    mu_tilde[reduce_index(i, j)] = (omega[i] * mu[i] + omega[j] * mu[j]) / (omega[i] + omega[j]) - (epsilon * omega[j]) / (omega[i] + omega[j])
    return copy(mu_tilde)
end

##################################################

function reduce_laplacian(laplacian::Array{<:Real}, i::Int64, j::Int64)::Array{Float64}
    # Function that produces a new (K-1)x(K-1) matrix from the original Laplacian
    # by adding ind_rem row/column to ind_keep row/column and adjusting diagonal 
    # entries accordingly

    laplacian_tilde = copy(laplacian)
    laplacian_tilde[i,:] += laplacian_tilde[j,:]
    laplacian_tilde[:,i] += laplacian_tilde[:,j]
    laplacian_tilde = laplacian_tilde[[1:(j - 1); (j + 1):end], [1:(j - 1); (j + 1):end]]
    # laplacian_tilde[reduce_index(i, j), reduce_index(i, j)] -= sum(laplacian_tilde[reduce_index(i, j), :])
    return copy(laplacian_tilde)
end

##################################################

function reduce_laplacian_j(laplacian::Array{<:Real}, i::Int64, j::Int64)::Array{Float64}
    laplacian_tilde_j = laplacian[:, j]
    laplacian_tilde_j[j] += laplacian_tilde_j[i]
    laplacian_tilde_j = laplacian_tilde_j[[(1:i - 1); (i + 1):end]]
    return copy(laplacian_tilde_j)
end

##################################################

function response_vanilla(setting::Setting, i::Int64, j::Int64)::Array{Float64}
    lambda_response = copy(setting.mu)
    lambda_response[i] = (setting.mu[i] * setting.omega[i] + setting.mu[j] * setting.omega[j] - setting.epsilon * setting.omega[j]) / (setting.omega[i] + setting.omega[j])
    lambda_response[j] = lambda_response[i] + setting.epsilon
    return copy(lambda_response)
end

##################################################

function response_spectral(setting::Setting, i::Int64, j::Int64)::Array{Float64}
    lambda_vanilla = response_vanilla(setting, i, j)
    if smoothness(lambda_vanilla, setting.laplacian) <= setting.R
        # println("Vanilla oracle is good ehough with smoothness $(smoothness(lambda_vanilla, setting.laplacian))")
        return lambda_vanilla
    end

    gamma_low = 0
    gamma_up = 0.0001
    precision = 1e-14
    lambda_spectral = lambda_vanilla

    while (smoothness(response_gamma(setting, i, j, gamma_up), setting.laplacian) > setting.R)
        gamma_up *= 2
        if gamma_up >= 1e5
            println("-"^50)
            println("Warning:")
            println("Gamma is getting out of control.")
            println("Probably too large value of epsilon or small value of R")
            println("gamma = $gamma_up  ->  R = $(smoothness(response_gamma(setting, i, j, gamma_up), setting.laplacian))")
            println("-"^50)
        end
    end

    # println("gamma_low = $gamma_low")
    # println("gamma_up = $gamma_up")
    
    while abs(gamma_up - gamma_low) > precision
        gamma_mid = (gamma_up + gamma_low) / 2
        R = smoothness(response_gamma(setting, i, j, gamma_mid), setting.laplacian)
        if R > setting.R
            gamma_low = gamma_mid
        else
            gamma_up = gamma_mid
        end
    end

    # println("gamma_low = $gamma_low")
    # println("gamma_up = $gamma_up")

    lambda = response_gamma(setting, i, j, gamma_up)
    
    # R = smoothness(lambda, bandit.laplacian)
    # println("lambda = $lambda")
    # println("R = $R")

    return lambda
end

##################################################

function response_gamma(setting::Setting, i::Int64, j::Int64, gamma::Real)::Array{Float64}
    O_t = Diagonal(reduce_omega(setting.omega, i, j))
    L_t = reduce_laplacian(setting.laplacian, i, j)
    L_t_j = reduce_laplacian_j(setting.laplacian, i, j)
    mu_t = reduce_mu(setting.mu, setting.omega, i, j, setting.epsilon)

    A = (O_t + 2gamma * L_t)
    lambda_gamma = inv(A' * A) * A' * (O_t * mu_t - 2gamma * setting.epsilon * L_t_j)  # changed to - before last term, check if it is correct at the moment
    lambda = recover_lambda(lambda_gamma, i, j, setting.epsilon)
    return copy(lambda)
end

##################################################

function smallest_R(setting::Setting)::Float64
    R = Inf
    I = [i for i in 1:length(setting.mu) if (setting.mu[i] >= maximum(setting.mu) - setting.epsilon)]
    # println("I = $I")
    gamma = 1e5

    for i in I, j in [k for k in 1:length(setting.mu) if k != i]
        lambda = response_gamma(setting, i, j, gamma)
        R_temp = smoothness(lambda, setting.laplacian)
        # println("i = $i, j = $j, R_temp = $R_temp")
        if R_temp < R
            R = R_temp
        end
    end

    return R
end

##################################################

function response_oracle_i(setting::Setting, i::Int64)::Array{Float64}
    (setting.R == Inf || norm(abs.(setting.laplacian)) < 1e-10 ) ? is_vanilla = true : is_vanilla = false

    response_objective = Inf
    response_lambda = zeros(length(setting.mu))

    for j in [k for k in 1:length(setting.mu) if k != i]
        if is_vanilla
            temp_lambda = response_vanilla(setting, i, j)
        else
            temp_lambda = response_spectral(setting, i, j)
        end
        temp_objective = objective(setting, temp_lambda)
        if temp_objective < response_objective
            response_objective = temp_objective
            response_lambda = temp_lambda
        end
    end
    return response_lambda
end

##################################################

function response_oracle(setting::Setting, i::Int64)::Array{Float64}
    return response_oracle_i(setting, i)
end

##################################################

function response_oracle(setting::Setting)::Array{Float64}
    I = [k for k in 1:length(setting.mu) if (setting.mu[k] >= maximum(setting.mu) - setting.epsilon)]

    response_objective = Inf
    response_lambda = zeros(setting.K)

    for i in I
        temp_lambda = response_oracle_i(setting, i)
        temp_objective = objective(setting, temp_lambda)
        if temp_objective < response_objective
            response_objective = temp_objective
            response_lambda = temp_lambda
        end
    end
    return response_lambda
end

##################################################

function gradient_omega_i(setting, i)
    lambda = response_oracle(setting, i)
    grad = (lambda - setting.mu).^2 / 2
    # grad -= sum(grad) * ones(setting.K) / setting.K
    return grad
end

##################################################

function gradient_omega(setting)
    lambda = response_oracle(setting)
    grad = (lambda - setting.mu).^2 / 2
    return grad
end

##################################################

function line_graph_laplacian(N)
    laplacian = zeros(N,N)
    for i = 1:(N-1)
        laplacian[i+1,i] -= 1
        laplacian[i,i+1] -= 1
        laplacian[i,i] += 1
        laplacian[i+1,i+1] += 1
    end
    return laplacian
end

##################################################

# function plot_distribution(setting::Setting)
#     K = setting.K
  
#     X = 0:0.01:1
#     Y = setting.f.(X)
#     scale = ((K*(maximum(setting.omega)+0.01))/maximum(Y))
#     Y = Y .* scale
#     m = maximum(Y)
  
#     # plt = plot(X,Y,color=:black,legend=false, ylims=(0,13))
#     plt = plot(X,Y,color=:black,legend=false)
  
#     # plot!([0,1], [m-setting.epsilon*scale, m-setting.epsilon*scale], legend=false, color=:red)
  
#     for i = 1:K
#       X = [(i-1)*(1/K), (i)*(1/K)]
#       Y = [K*setting.omega[i], K*setting.omega[i]]
#       setting.mu[i] >= maximum(setting.mu)-setting.epsilon ? c = :red : c = :blue
#       if setting.mu[i] >= maximum(setting.mu)-2*setting.epsilon && setting.mu[i] < maximum(setting.mu)-setting.epsilon
#         c = :green
#       end
#       plot!(X, Y, legend=false, color=:black, fill=(0,0.5,c))
#     end
  
  
#     X = zeros(2 * setting.K)
#     Y = zeros(2 * setting.K)
#     for i = 1:setting.K
#       # println((i-1) / setting.K)
#       X[2i - 1] = (i-1) / setting.K
#       Y[2i - 1] = setting.omega[i] * setting.K
#       X[2i] = i / setting.K
#       Y[2i] = setting.omega[i] * setting.K
#     end
  
#     X = vcat(0,X,1)
#     Y = vcat(0,Y,0)
  
#     # # println("X = $X")
#     # # println("Y = $Y")
  
#     # display(plot(X,Y))
#     xt = ([i/setting.K - 1/(2setting.K) for i=1:setting.K],collect(1:setting.K))
#     xt = 0:0.1:64
#     # xt=false
#     plot!(X,Y, xticks=xt, xlims=(0,1), legend=false,color=:black)
    
#     display(plt)
#   end

##################################################

# function omega_star(setting; method="mirror", history=false, precision=1e-5, eta=0.01, T=100, C::Float64=1.0, tolerance::Int=10, error::Float64=1e-6, grad_ratio::Float64=1.0)
#     omegas = []
#     objectives = Float64[]
#     old_objective = objective(setting, lambda_opt(setting, setting.R))
#     append!(objectives, old_objective)
#     old_omega = copy(setting.omega)
#     current_tolerance = 0
  
#     function print_progress(t, T, N)
#       if t % round(T/N) == 0
#         println("Progress = $(round(100t/T))%")
#       end
#     end
  
#     # Mirror ascent version
  
    
#     # mirror ascent aclgorithm
  
#     # L = (maximum(setting.mu) - minimum(setting.mu))^2 / 2
#     # T = ceil(2log(setting.K) * (L/precision)^2)
#     # eta = sqrt((2log(setting.K))/(T*L^2))
    
  
#     println("Running gradient-based algorithm")
#     println("T = $T")
#     println("eta = $eta")
  
#     for t = 1:T
#         print_progress(t,T,100)
#         grad = omega_gradient(setting)
      
#         new_omega = []
#         if method == "mirror"
#             # println(grad)
#             new_omega = setting.omega .* exp.(eta * grad)
#             new_omega /= sum(new_omega)
#         elseif method == "gd"
#             new_omega = setting.omega + eta * grad
#             new_omega /= sum(new_omega)
#         end
  
#         for i = 1:setting.K
#             setting.omega[i] = new_omega[i]
#         end
  
#         obj = objective(setting, lambda_opt(setting, setting.R))

#         # println("----------")
#         # println(setting.omega)
#         # println(grad)

#         append!(objectives, obj)

#         if obj < old_objective
#             current_tolerance += 1
#         else
#             current_tolerance = 0
#             old_omega = copy(setting.omega)
#             old_objective = obj
#         end

#         if (current_tolerance == tolerance) || (t == T && current_tolerance > 0)
#             for i = 1:setting.K
#             setting.omega[i] = old_omega[i]
#             end
#             # objectives = objectives[1:end-tolerance]
#             println("----- Emergency stop: algorithm diverges -----")
#             break
#         end
#     end
    
#     return objectives
#   end

##################################################
end

##################################################
################## End of module #################
##################################################







##### testing module during coding
##### everything helow this point should be erased


using Plots

let test_cote = 1
    mu = [0.9; 0.5; 0.6]
    epsilon = 0.05
    laplacian = [0 0 0; 0 1 -1; 0 -1 1]
    R = 0.2
    
    S = BAI.create_setting(mu, epsilon, laplacian, R)

    lambda = BAI.response_oracle(S)
    obj = BAI.objective(S, lambda)

    println(obj)

    n_iter = 20000
    objs = zeros(n_iter)

    lr = 0.01

    for iter = 1:n_iter
        lambda = BAI.response_oracle(S)
        gradient = BAI.gradient_omega(S)
        objs[iter] = BAI.objective(S, lambda)

        S.omega = S.omega + lr * gradient
        S.omega = S.omega / sum(S.omega)
    end

    # println(objs)
    println("omega = $(S.omega)")
    display(plot(objs))

    # BAI.plot_distribution(S)



end



