# test of gradient calculation

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra, ForwardDiff
using Test

function test_gradient_sphere(
        dim::Int
    )
    # define a quadratic ideal for testing gradient calculation
    ideal = LowRankSOS.QuadraticIdeal(dim, [diagm([-1.0; ones(dim-1)])])
    # get the projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal)
    # set the target rank
    rank = dim-1
    # define the target quadratic form and the starting point
    mat_target = zeros(dim,dim) # diagm(Float64.(collect(1:dim)))
    mat_start = ones(rank,dim)
    mat_Gram_diff = mat_start'*mat_start - mat_target
    # calculate the differentials
    mat_grad = LowRankSOS.compute_obj_grad(mat_start, mat_target, map_quotient)
    func_obj_val = (mat_temp)->LinearAlgebra.norm(LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_target - mat_temp'*mat_temp), dim=dim))^2
    mat_grad_auto = ForwardDiff.gradient(func_obj_val, mat_start)
    return mat_grad ≈ mat_grad_auto
end

function test_gradient_ternary_quartics()
    # define the ideal of embedded ternary quartics
    dim = 6
    # the entries correspond to monomials
    # x², xy, xz, y², yz, z²
    # there are 6 generators in the ideal
    mat_gen = Vector{Matrix{Float64}}(undef, 6)
    # generator 1: x²⋅y² - (xy)²
    mat_gen[1] = zeros(dim,dim)
    mat_gen[1][1,4] = 1/2
    mat_gen[1][4,1] = 1/2
    mat_gen[1][2,2] = -1
    # generator 2: x²⋅z² - (xz)²
    mat_gen[2] = zeros(dim,dim)
    mat_gen[2][1,6] = 1/2
    mat_gen[2][6,1] = 1/2
    mat_gen[2][3,3] = -1
    # generator 3: y²⋅z² - (yz)²
    mat_gen[3] = zeros(dim,dim)
    mat_gen[3][4,6] = 1/2
    mat_gen[3][6,4] = 1/2
    mat_gen[3][5,5] = -1
    # generator 4: xy⋅xz - x²⋅yz
    mat_gen[4] = zeros(dim,dim)
    mat_gen[4][2,3] = 1/2
    mat_gen[4][3,2] = 1/2
    mat_gen[4][1,5] = -1/2
    mat_gen[4][5,1] = -1/2
    # generator 5: xy⋅yz - y²⋅xz
    mat_gen[5] = zeros(dim,dim)
    mat_gen[5][2,5] = 1/2
    mat_gen[5][5,2] = 1/2
    mat_gen[5][4,3] = -1/2
    mat_gen[5][3,4] = -1/2
    # generator 6: xz⋅yz - z²⋅xy
    mat_gen[6] = zeros(dim,dim)
    mat_gen[6][3,5] = 1/2
    mat_gen[6][5,3] = 1/2
    mat_gen[6][2,6] = -1/2
    mat_gen[6][6,2] = -1/2
    # define the ideal
    ideal_ternary_quartics = LowRankSOS.QuadraticIdeal(dim, mat_gen)
    # get the projection matrix associated with the canonical quotient map
    map_quotient = LowRankSOS.construct_quotient_map(ideal_ternary_quartics)
    # set the target rank
    rank = 3
    # choose a target quadratic form corresponding to x⁴+x²y²+y⁴+z⁴
    mat_target = zeros(dim,dim)
    mat_target[1,1] = 1
    mat_target[2,2] = 1
    mat_target[4,4] = 1
    mat_target[6,6] = 1
    # choose a starting point
    mat_start = [1.0 0.0 0.0 0.0 0.0 0.0;
                 0.0 10.0 0.0 0.0 0.0 0.0;
                 0.0 0.0 0.0 1.0 0.0 0.0]
    # calculate the differentials
    mat_grad = LowRankSOS.compute_obj_grad(mat_start, mat_target, map_quotient)
    func_obj_val = (mat_temp)->LinearAlgebra.norm(LowRankSOS.convert_vec_to_sym(map_quotient * LowRankSOS.convert_sym_to_vec(mat_target - mat_temp'*mat_temp), dim=dim))^2
    mat_grad_auto = ForwardDiff.gradient(func_obj_val, mat_start)
    return mat_grad ≈ mat_grad_auto
end

@test test_gradient_sphere(5)
@test test_gradient_ternary_quartics()

