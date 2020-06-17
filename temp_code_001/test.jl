using DelimitedFiles
using SparseArrays
using LowRankModels
using Random
using JLD
using Plots

function remove_negligible_entries(data, entry_type::String, n_ratings::Int)
    # entry_type is either "movies" or "users".
    # keed every movie/user with at least n_ratings ratings
    if entry_type == "movies"
        column = 2
    elseif entry_type == "users"
        column = 1
    else
        column = 0
        println("Unknown entry_type\n -- Use \"movies\" to remove movies with small number ratings\n -- Use \"users\" to remove users with small number of ratings")
        return copy(data)
    end

    new_data = []
    movies_keep = Set()
    rows_keep = []

    for id in Set(data[:, column])
        if sum(data[:, column] .== id) >= n_ratings
            push!(movies_keep, id)
        end
    end

    for i in 1:length(data[:, column])
        if data[i, column] in movies_keep
            push!(rows_keep, i)
        end
    end

    new_data = data[rows_keep, :]

    return new_data
end

function print_number_of_entries(data, entry_type::String)
    if entry_type == "movies"
        column = 2
    elseif entry_type == "users"
        column = 1
    else
        column = 0
        println("Unknown entry_type\n -- Use \"movies\" to remove movies with small number ratings\n -- Use \"users\" to remove users with small number of ratings")
        return nothing
    end

    println("Number os $entry_type is $(length(Set(data[:, column])))")
end

function get_number_of_entries(data, entry_type::String)
    if entry_type == "movies"
        column = 2
    elseif entry_type == "users"
        column = 1
    else
        column = 0
        println("Unknown entry_type\n -- Use \"movies\" to remove movies with small number ratings\n -- Use \"users\" to remove users with small number of ratings")
        return 0
    end

    return length(Set(data[:, column]))
end

function reindex_matrix(matrix)
    (I, J, V) = findnz(matrix)
    I_map = sort(collect(Set(I)))
    J_map = sort(collect(Set(J)))

    I_new = map(i -> findfirst(I_map.==i), I)
    J_new = map(j -> findfirst(J_map.==j), J)

    return sparse(I_new, J_new, V)

end

function completion_error(A_original, A_new)
    (I, J, V) = findnz(A_original)

    error_number = 0
    error_size = 0

    for i in 1:length(I)
        a = A_original[I[i], J[i]]
        b = A_new[I[i], J[i]]
        if a != b
            error_number += 1
            error_size += abs(a-b)
        end
    end

    return (error_number, error_size)

end

function split_data(A; ratio=0.8)
    N::Int = round(ratio * size(A)[1])
    perm = Random.randperm(size(A)[1])

    graph_indices = perm[1:N]
    bandit_indices = perm[(N+1):end]

    ratings_graph = A[graph_indices, :]
    ratings_bandit_problems = A[bandit_indices, :]

    return (ratings_graph, ratings_bandit_problems)
end



function get_graph_from_data(A; k=5)
    N = size(A)[2]
    if k > N
        k = N
    end
    D = zeros(N,N)
    for i = 1:N, j = 1:N
        a = A[:, i]
        b = A[:, j]
        D[i, j] = sqrt(sum((a-b).^2))
        if i == j
            D[i, j] = -1
        end
    end

    laplacian = zeros(N,N)

    for i = 1:N
        perm = sortperm(D[i, :])[1:(k+1)]
        laplacian[i, perm] .= -1
        laplacian[perm, i] .= -1
    end

    for i = 1:N
        laplacian[i, i] -= sum(laplacian[i, :])
    end

    return laplacian
end


let main_code = 1
    data = readdlm("dataset/u.data", '\t', Int, '\n')

    data = remove_negligible_entries(data, "movies", 200)
    data = remove_negligible_entries(data, "users", 50)
    
    print_number_of_entries(data, "movies")
    print_number_of_entries(data, "users")

    data_matrix = sparse(data[:,1], data[:,2], data[:,3])

    size(data_matrix)

    data_matrix = reindex_matrix(data_matrix)

    size(data_matrix)

    glrm = GLRM(data_matrix, QuadLoss(), ZeroReg(), ZeroReg(), 50)


    for i = 1:100
        X, Y, ch = fit!(glrm, niter=1000)
    end

    X, Y, ch = fit!(glrm, niter=1000)

    ch.objective

    data_matrix
    X'Y

    minimum(data_matrix)
    maximum(data_matrix)

    minimum(X'Y)
    maximum(X'Y)

    
    for i = 1:size(data_matrix)[2]
        print("$i    ")
        print(sum(round.(X'Y)[:,i] .> 5))
        print("    ")
        print(sum(round.(X'Y)[:,i] .< 0))
        println()
    end



    completion_error(data_matrix, X'Y)
    completion_error(data_matrix, round.(X'Y))
    
    ratings = X'Y
    ratings = round.(ratings)
    ratings[ ratings .<= 0 ] .= 0
    ratings[ ratings .>= 5 ] .= 5

    size(ratings)

    minimum(ratings)
    maximum(ratings)

    ratings
    (ratings_graph, ratings_bandit_problems) = split_data(ratings)
    ratings_graph
    ratings_bandit_problems

    r = ratings_bandit_problems[6,:]
    h = histogram(r, bins=6)
    
    display(h)

    for i = 0:5
        println(sum(r .== i))
    end

    L = get_graph_from_data(ratings_graph)

    mu = ratings_bandit_problems[1,:]
    # mu = rand(0:5, 118)
    mu' * L * mu

    save("graph.jld", "laplacian", L)
    save("bandits.jld", "bandits", ratings_bandit_problems)

end 