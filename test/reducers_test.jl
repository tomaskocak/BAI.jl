using BAI
using Test

@test BAI.reduce_index(10, 3) == 9
@test BAI.reduce_index(10, 11) == 10
@test BAI.recover_index(10, 3) == 11
@test BAI.recover_index(10, 10) == 11
@test BAI.recover_index(10, 11) == 10