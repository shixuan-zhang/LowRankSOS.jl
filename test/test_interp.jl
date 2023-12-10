# test of interpolation of a quartic polynomial (for step selection)

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra, Test

# function that evaluates a univariate quartic polynomial with the form f(x)=a₄x⁴+a₃x³+a₂x²+a₁x
function eval_quartic(a,t)
    return a'*(t.^[1,2,3,4])
end

# function that tests whether the quartic polynomial is correctly interpolated
function test_interpoloation()
    # test the polynomial with coefficients [1,2,3,4] at values [-2,-1,1,2]
    let a = [1,2,3,4], x = [-2,-1,1,2]
        v = (t->eval_quartic(a,t)).(x)
        @test LowRankSOS.interpolate_quartic_polynomial(x,v) == a
    end
    # test the polynomial with coefficients [4,3,2,1] at values [-1,-1/3,1/3,1]
    let a = [4,3,2,1], x = [-1,-1/3,1/3,1]
        v = (t->eval_quartic(a,t)).(x)
        @test LowRankSOS.interpolate_quartic_polynomial(x,v) ≈ a
    end
end

test_interpoloation()

