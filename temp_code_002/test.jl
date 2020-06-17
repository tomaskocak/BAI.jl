using JLD
using LinearAlgebra

include("src/BAI.jl")



let main_code = 1

    K = 10
    
    # L = load("temp_code_001/graph.jld")["laplacian"]
    # ratings = load("temp_code_001/bandits.jld")["bandits"]

    mu = rand(0:5, K)
    epsilon = 0.05
    laplacian = zeros(K,K)

    for i = 1:K
        edges = ((mu .== mu[i]) + (mu .== (mu[i]+1)) + (mu .== (mu[i]-1))) .== 1
        laplacian[i, edges] .= -1
        laplacian[edges, i] .= -1
    end

    laplacian = laplacian - Diagonal(sum(laplacian, dims=1)[:])
    L = laplacian

    R = mu' * L * mu

    # m = rand(0:5, K)
    # m' * L * m
    
    S = BAI.create_setting(mu, epsilon, laplacian, R)

    lambda = BAI.response_oracle(S)
    obj = BAI.objective(S, lambda)

    println(obj)

    n_iter = 100
    objs = zeros(n_iter)

    lr = 0.01

    for iter = 1:n_iter
        lambda = BAI.response_oracle(S)
        gradient = BAI.gradient_omega(S)
        objs[iter] = BAI.objective(S, lambda)

        S.omega = S.omega + lr * gradient
        S.omega = S.omega / sum(S.omega)

        println("Iteration: $iter      objective: $(objs[iter])")
    end

    # println(objs)
    println("omega = $(S.omega)")
    display(plot(objs))


end


