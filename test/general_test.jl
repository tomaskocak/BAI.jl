# using BAI
# using Test
# using LinearAlgebra

tolerance = 1e-13

mu = [1.2; 3.5; 1.1; 0.4; 2.9]
omega = [0.1; 0.2; 0.4; 0.1; 0.2]
lambda = [2.3; 7.1; 2.4; 0.2; 7.1]
laplacian = zeros(Float64, 5, 5)
for i in 1:length(lambda), j in (i + 1):length(lambda)
    w = rand()
    laplacian[i, j], laplacian[j, i] = -w, -w
end
for i in 1:length(lambda)
    laplacian[i, i] -= sum(laplacian[:, i])
end
smoothness = 0
for i in 1:length(lambda), j in 1:length(lambda)
    if i != j
        global smoothness += -laplacian[i, j] * (lambda[i] - lambda[j])^2 / 2
    end
end
objective = 0
for i in 1:length(lambda)
    global objective += omega[i] * (mu[i] - lambda[i])^2 / 2
end
@test norm(BAI.smoothness(lambda, laplacian) - smoothness) < tolerance
@test norm(BAI.objective(mu, omega, lambda) - objective) < tolerance

B = BAI.Bandit([1; 2; 4], 0.0, zeros(3, 3), 0.0)
BA = BAI.create_bandit_problem([1; 2; 4])
BB = BAI.create_bandit_problem([1; 2; 4], 0)
BC = BAI.create_bandit_problem([1; 2; 4], 0, zeros(3, 3), 0)

@test B.mu == BA.mu && B.mu == BB.mu && B.mu == BC.mu 
@test B.epsilon == BA.epsilon && B.epsilon == BB.epsilon && B.epsilon == BC.epsilon
@test B.laplacian == BA.laplacian && B.laplacian == BB.laplacian && B.laplacian == BC.laplacian
@test B.R == BA.R && B.R == BB.R && B.R == BC.R
