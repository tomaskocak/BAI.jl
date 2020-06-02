using BAI
using Test
using LinearAlgebra

tolerance = 1e-14

@test BAI.reduce_index(10, 3) == 9
@test BAI.reduce_index(10, 11) == 10
@test BAI.recover_index(10, 3) == 11
@test BAI.recover_index(10, 10) == 11
@test BAI.recover_index(10, 11) == 10


# testing reduce_omega
omega = [7, 13, 2, 7, 12]
i, j = 3, 1
omega_tilde = [13, 9, 7, 12]
@test BAI.reduce_omega(omega, i, j) == omega_tilde

# testing reduce_lambda
lambda = [1.2, 3.5, 1.123, 7.32, 10.23, 1.5]
j = 4
lambda_tilde = [1.2, 3.5, 1.123, 10.23, 1.5]
@test BAI.reduce_lambda(lambda, j) == lambda_tilde

# testing recover_lambda
lambda_tilde = [2.3, 7.41, 9.2, 10, 13.5]
i, j = 4, 2
epsilon = 0.01
lambda = [2.3, 9.21, 7.41, 9.2, 10, 13.5]
@test norm(BAI.recover_lambda(lambda_tilde, i, j, epsilon) - lambda) < tolerance

# testing reduce_mu
mu = [1.2, 3.4, 5.6, 7.8]
omega = [0.1, 0.4, 0.3, 0.2]
i, j = 2, 3
epsilon = 0.01
mu_tilde = [1.2, 4.337142857142857, 7.8]
@test norm(BAI.reduce_mu(mu, omega, i, j, epsilon) - mu_tilde) < tolerance

# testing reduce_laplacian
laplacian = [2 -1 -1; -1 1.5 -0.5; -1 -0.5 1.5]
i, j = 2, 1
laplacian_tilde = [1.5 -1.5; -1.5 1.5]
@test BAI.reduce_laplacian(laplacian, i, j) == laplacian_tilde

# testing reduce_laplacian_j
laplacian = [2 -1 -1; -1 1.5 -0.5; -1 -0.5 1.5]
i, j = 2, 1
laplacian_tilde_j = [1; -1]
@test BAI.reduce_laplacian_j(laplacian, i, j) == laplacian_tilde_j
