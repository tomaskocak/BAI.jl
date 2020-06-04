# using BAI
# using Test
# using LinearAlgebra


mu = [1; 0.9; 0.6]
omega = [0.3; 0.4; 0.4]
laplacian = [2 -1 -1; -1 1.5 -0.5; -1 -0.5 1.5]
i, j = 1, 2
laplacian_tilde = [1.5 -1.5; -1.5 1.5]
@test BAI.reduce_laplacian(laplacian, i, j) == laplacian_tilde

# B = BAI.create_bandit_problem(mu, 0.2, laplacian, 0.1)

# BAI.response_spectral(B, omega, i, j)



# # epsilons = 0:10:200
# # Rs = zeros(length(epsilons))

# # for k = 1:length(epsilons)
# #     B.epsilon = epsilons[k]
# #     Rs[k] = BAI.smallest_R(B, omega)
# #     println("epsilon = $(epsilons[k]), R = $(Rs[k])")
# # end