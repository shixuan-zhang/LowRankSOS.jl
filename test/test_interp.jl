# test of interpolation of a quartic polynomial (for step selection)

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra, Polynomials, Test

# function that tests whether the quartic polynomial is correctly interpolated
function test_interpoloation()
    # test the polynomial with coefficients [1,2,3,4]
    let a = [1,2,3,4]
        p = Polynomial([0;a])
        let x = [-1,-0.1,0.1,1]
            @test LowRankSOS.interpolate_quartic_Vandermonde(x,p.(x)) ≈ a
        end
        let x = LowRankSOS.Chebx
            @test LowRankSOS.interpolate_quartic_Chebyshev(p.(x)) ≈ [0;a]
        end
    end
    # test the polynomial with coefficients [4,3,2,1] at values [-1,-1/3,1/3,1]
    let a = [4,3,2,1]
        p = Polynomial([0;a])
        let x = [-1,-1/3,1/3,1]
            @test LowRankSOS.interpolate_quartic_Vandermonde(x,p.(x)) ≈ a
        end
        let x = LowRankSOS.Chebx
            @test LowRankSOS.interpolate_quartic_Chebyshev(p.(x)) ≈ [0;a]
        end
    end
end

test_interpoloation()

