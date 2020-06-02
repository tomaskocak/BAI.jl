function reduce_index(a, j)
    return a < j ? a : a - 1
end

function recover_index(a, j)
    return a < j ? a : a + 1
end


println(reduce_index(10, 3))
println(reduce_index(10, 11))
println(recover_index(10, 3))
println(recover_index(10, 10))
println(recover_index(10, 11))


function reduce_omega(omega, i, j)
    # new_omega = zeros
end