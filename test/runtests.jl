using BAI
using LinearAlgebra
using Test
using Plots


# @testset "All tests" begin
#     @testset "general" begin
#         include("general_test.jl")
#     end

#     @testset "reducers" begin
#         include("reducers_test.jl")
#     end

#     @testset "response oracle" begin
#         include("response_oracle_test.jl")
#     end
# end


##### random code that needs to be deleted later


let main_code = 1
    mu = [0.9; 0.5; 0.6]
    laplacian = [0 0 0; 0 1 -1; 0 -1 1]
    epsilon = 0
    R = 0.01

    B = BAI.create_bandit_problem(mu, epsilon, laplacian, R)

    omega = [1.0, 1.0, 1.0]./3

    lambda = BAI.response_oracle(B, omega)
    obj = BAI.objective(mu, omega, lambda)

    println(obj)

    n_iter = 100
    objs = zeros(n_iter)

    lr = 0.1

    for iter = 1:n_iter
        lambda = BAI.response_oracle(B, omega)
        objs[iter] = BAI.objective(B.mu, omega, lambda)

        omega = omega + lr * lambda
        omega = omega / sum(omega)
    end

#    println(objs)
    println("omega = $omega")
    display(plot(objs))
end





# BAI.response_spectral(B, omega, i, j)



# # epsilons = 0:10:200
# # Rs = zeros(length(epsilons))

# # for k = 1:length(epsilons)
# #     B.epsilon = epsilons[k]
# #     Rs[k] = BAI.smallest_R(B, omega)
# #     println("epsilon = $(epsilons[k]), R = $(Rs[k])")
# # end



