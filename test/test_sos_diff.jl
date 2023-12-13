# test of Jacobian and gradient calculation

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra, SparseArrays, ForwardDiff, Test

## test the sos map and its differential (Jacobian matrix) using a (2,2)-scroll
# linear forms on the two-dimensional scroll correspond to bihomogeneous polynomials in x,y,s,t
# the entries correspond to monomials x⋅s², x⋅s⋅t, x⋅t², y⋅s², y⋅s⋅t, y⋅t²
# the quadratic forms are spanned by the monomials
# x²⋅s⁴,    x²⋅s³⋅t,    x²⋅s²⋅t²,   x²⋅s⋅t³,    x²⋅t⁴,
# x⋅y⋅s⁴,   x⋅y⋅s³⋅t,   x⋅y⋅s²⋅t²,  x⋅y⋅s⋅t³,   x⋅y⋅t⁴,
# y²⋅s⁴,    y²⋅s³⋅t,    y²⋅s²⋅t²,   y²⋅s⋅t³,    y²⋅t⁴

function customized_diff_map(
        L::Vector
    )
    M = zeros(typeof(L[1]),15,18)
    for i = 0:2
        M[:,1+6*i] = [L[1+6*i],L[2+6*i],L[3+6*i],0,0,
                      L[4+6*i],L[5+6*i],L[6+6*i],0,0,
                      0,0,0,0,0]
        M[:,2+6*i] = [0,L[1+6*i],L[2+6*i],L[3+6*i],0,
                      0,L[4+6*i],L[5+6*i],L[6+6*i],0,
                      0,0,0,0,0]
        M[:,3+6*i] = [0,0,L[1+6*i],L[2+6*i],L[3+6*i],
                      0,0,L[4+6*i],L[5+6*i],L[6+6*i],
                      0,0,0,0,0]
        M[:,4+6*i] = [0,0,0,0,0,
                      L[1+6*i],L[2+6*i],L[3+6*i],0,0,
                      L[4+6*i],L[5+6*i],L[6+6*i],0,0]
        M[:,5+6*i] = [0,0,0,0,0,
                      0,L[1+6*i],L[2+6*i],L[3+6*i],0,
                      0,L[4+6*i],L[5+6*i],L[6+6*i],0]
        M[:,6+6*i] = [0,0,0,0,0,
                      0,0,L[1+6*i],L[2+6*i],L[3+6*i],
                      0,0,L[4+6*i],L[5+6*i],L[6+6*i]]
    end
    return 2*M
end

# define the sum-of-square map manually
function customized_sos_map(
        L::Vector
    )
    return [sum(L[i*6+1]^2 for i=0:2),
            2*sum(L[i*6+1]*L[i*6+2] for i=0:2),
            2*sum(L[i*6+1]*L[i*6+3] for i=0:2) + sum(L[i*6+2]^2 for i=0:2),
            2*sum(L[i*6+2]*L[i*6+3] for i=0:2),
            sum(L[i*6+3]^2 for i=0:2),
            2*sum(L[i*6+1]*L[i*6+4] for i=0:2),
            2*sum(L[i*6+1]*L[i*6+5] + L[i*6+2]*L[i*6+4] for i=0:2),
            2*sum(L[i*6+1]*L[i*6+6] + L[i*6+3]*L[i*6+4] + L[i*6+2]*L[i*6+5] for i=0:2),
            2*sum(L[i*6+3]*L[i*6+5] + L[i*6+2]*L[i*6+6] for i=0:2),
            2*sum(L[i*6+3]*L[i*6+6] for i=0:2),
            sum(L[i*6+4]^2 for i=0:2),
            2*sum(L[i*6+4]*L[i*6+5] for i=0:2),
            2*sum(L[i*6+4]*L[i*6+6] for i=0:2) + sum(L[i*6+5]^2 for i=0:2),
            2*sum(L[i*6+5]*L[i*6+6] for i=0:2),
            sum(L[i*6+6]^2 for i=0:2)
           ]
end


# function that tests whether the sos map and its differential are calculated correctly
function test_sos_diff()
    # use the monomial multi-indices to create the multiplication table
    m1 = [[1,0,2,0],[1,0,1,1],[1,0,0,2],[0,1,2,0],[0,1,1,1],[0,1,0,2]]
    m2 = [[2,0,4,0],[2,0,3,1],[2,0,2,2],[2,0,1,3],[2,0,0,4],
          [1,1,4,0],[1,1,3,1],[1,1,2,2],[1,1,1,3],[1,1,0,4],
          [0,2,4,0],[0,2,3,1],[0,2,2,2],[0,2,1,3],[0,2,0,4]]
    # create a dictionary to look up the deg-2 monomial indices
    d2 = Dict(m2[i] => i for i in 1:15)
    # define the coordinate ring by the products
    p = SparseVector{Rational{Int},Int}[]
    for j = 1:6
        for i = 1:j
            i2 = d2[m1[i]+m1[j]]
            push!(p, sparsevec([i2], [1], 15))
        end
    end
    R2 = CoordinateRing2(6,15,p)
    # test at the forms (x⋅s², x⋅s⋅t, x⋅t²)
    let L = [1,0,0,0,0,0,
             0,1,0,0,0,0,
             0,0,1,0,0,0]
        @test get_sos(L,R2) == customized_sos_map(L)
        @test collect(build_Jac_mat(L,R2)) == customized_diff_map(L)
        @test ForwardDiff.jacobian(customized_sos_map,L) == collect(build_Jac_mat(L,R2))
        # test the differential calculation at a target x²⋅s⁴ + 2x²⋅s²⋅t² + x²⋅t⁴ + y²⋅t⁴
        let T = [1,0,2,0,1,
                 0,0,0,0,0,
                 0,0,0,0,1]
            customized_obj = (l)->norm(customized_sos_map(l)-T,2)^2
            # test the gradient
            @test ForwardDiff.gradient(customized_obj,L) ≈ 2*build_Jac_mat(L,R2)'*(get_sos(L,R2)-T)
            # test the Hessian
            @test ForwardDiff.hessian(customized_obj,L) ≈ build_Hess_mat(3,L,T,R2)
        end
        # test the differential calculation at a dense quadric target
        let T = [1,3,2,3,1,
                 5,4,5,2,1,
                 4,2,3,3,1]
            customized_obj = (l)->norm(customized_sos_map(l)-T,2)^2
            # test the gradient
            @test ForwardDiff.gradient(customized_obj,L) ≈ 2*build_Jac_mat(L,R2)'*(get_sos(L,R2)-T)
            # test the Hessian
            @test ForwardDiff.hessian(customized_obj,L) ≈ build_Hess_mat(3,L,T,R2)
        end
    end
    # test at the forms (x⋅s²-y⋅t², x⋅t²-y⋅s², √2(x⋅s⋅t+y⋅s⋅t))
    let L = [1,0,0,0,0,-1,
             0,0,1,-1,0,0,
             0,√2,0,0,√2,0]
        @test get_sos(L,R2) == customized_sos_map(L)
        @test collect(build_Jac_mat(L,R2)) == customized_diff_map(L)
        @test ForwardDiff.jacobian(customized_sos_map,L) == collect(build_Jac_mat(L,R2))
        # test the differential calculation at a target (x²+y²)(s²+t²)² + x⋅y⋅(s³⋅t+s⋅t³)
        let T = [1,0,2,0,1,
                 0,1,0,1,0,
                 1,0,2,0,1]
            customized_obj = (l)->norm(customized_sos_map(l)-T,2)^2
            # test the gradient
            @test ForwardDiff.gradient(customized_obj,L) ≈ 2*build_Jac_mat(L,R2)'*(get_sos(L,R2)-T)
            # test the Hessian
            @test ForwardDiff.hessian(customized_obj,L) ≈ build_Hess_mat(3,L,T,R2)
        end
        # test the differential calculation at a dense quadric target
        let T = [1,3,2,3,1,
                 5,4,5,2,1,
                 4,2,3,3,1]
            customized_obj = (l)->norm(customized_sos_map(l)-T,2)^2
            # test the gradient
            @test ForwardDiff.gradient(customized_obj,L) ≈ 2*build_Jac_mat(L,R2)'*(get_sos(L,R2)-T)
            # test the Hessian
            @test ForwardDiff.hessian(customized_obj,L) ≈ build_Hess_mat(3,L,T,R2)
        end
    end
    # test at a dense tuple of linear forms
    let L = [1,2,3,4,5,6,
             6,5,4,3,2,1,
             1,1,1,1,1,1]
        @test get_sos(L,R2) == customized_sos_map(L)
        @test collect(build_Jac_mat(L,R2)) == customized_diff_map(L)
        @test ForwardDiff.jacobian(customized_sos_map,L) == collect(build_Jac_mat(L,R2))
    end
end


test_sos_diff()
