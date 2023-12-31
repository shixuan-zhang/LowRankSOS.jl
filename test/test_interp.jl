# test of interpolation of a quartic polynomial (for step selection)

include("../src/LowRankSOS.jl")
using .LowRankSOS

using LinearAlgebra, SparseArrays, Polynomials, Test

## test the interpolation on line searches on the Veronese surface
# linear forms on the Veronese surface correspond to quadratic forms in x,y,z
# the entries correspond to monomials x², x⋅y, y², x⋅z, y⋅z, z² 
# the quadratic forms are spanned by the monomials
# x⁴,     x³⋅y,   x²⋅y²,  x⋅y³, y⁴,
# x³⋅z,   x²⋅y⋅z, x⋅y²⋅z, y³⋅z, x²⋅z², 
# x⋅y⋅z², y²⋅z²,  x⋅z³,   y⋅z³, z⁴

function customized_interp(
        L::Vector, # linear forms
        T::Vector, # target quadric
        D::Vector  # moving direction
    )
    # define the degree 0 part in terms of the step size s in the quadric difference
    d0 = [sum(L[6*i+1]^2 for i=0:2)-T[1],
          2*sum(L[6*i+1]*L[6*i+2] for i=0:2)-T[2],
          sum(L[6*i+2]^2 for i=0:2)+2*sum(L[6*i+1]*L[6*i+3] for i=0:2)-T[3],
          2*sum(L[6*i+2]*L[6*i+3] for i=0:2)-T[4],
          sum(L[6*i+3]^2 for i=0:2)-T[5],
          2*sum(L[6*i+1]*L[6*i+4] for i=0:2)-T[6],
          2*sum(L[6*i+2]*L[6*i+4] + L[6*i+1]*L[6*i+5] for i=0:2)-T[7],
          2*sum(L[6*i+2]*L[6*i+5] + L[6*i+3]*L[6*i+4] for i=0:2)-T[8],
          2*sum(L[6*i+3]*L[6*i+5] for i=0:2)-T[9],
          sum(L[6*i+4]^2 for i=0:2)+2*sum(L[6*i+1]*L[6*i+6] for i=0:2)-T[10],
          2*sum(L[6*i+2]*L[6*i+6] + L[6*i+4]*L[6*i+5] for i=0:2)-T[11],
          sum(L[6*i+5]^2 for i=0:2)+2*sum(L[6*i+3]*L[6*i+6] for i=0:2)-T[12],
          2*sum(L[6*i+4]*L[6*i+6] for i=0:2)-T[13],
          2*sum(L[6*i+5]*L[6*i+6] for i=0:2)-T[14],
          sum(L[6*i+6]^2 for i=0:2)-T[15]
         ]
    # define the degree 1 part in terms of the step size s in the quadric difference
    d1 = [sum(L[6*i+1]*D[6*i+1] for i=0:2),
          sum(L[6*i+1]*D[6*i+2] + D[6*i+1]*L[6*i+2] for i=0:2),
          sum(L[6*i+2]*D[6*i+2] + L[6*i+1]*D[6*i+3] + D[6*i+1]*L[6*i+3] for i=0:2),
          sum(L[6*i+2]*D[6*i+3] + D[6*i+2]*L[6*i+3] for i=0:2),
          sum(L[6*i+3]*D[6*i+3] for i=0:2),
          sum(L[6*i+1]*D[6*i+4] + D[6*i+1]*L[6*i+4] for i=0:2),
          sum(L[6*i+2]*D[6*i+4] + D[6*i+2]*L[6*i+4] + L[6*i+1]*D[6*i+5] + D[6*i+1]*L[6*i+5] for i=0:2),
          sum(L[6*i+2]*D[6*i+5] + D[6*i+2]*L[6*i+5] + L[6*i+3]*D[6*i+4] + D[6*i+3]*L[6*i+4] for i=0:2),
          sum(L[6*i+3]*D[6*i+5] + D[6*i+3]*L[6*i+5] for i=0:2),
          sum(L[6*i+4]*D[6*i+4] + L[6*i+1]*D[6*i+6] + D[6*i+1]*L[6*i+6] for i=0:2),
          sum(L[6*i+2]*D[6*i+6] + D[6*i+2]*L[6*i+6] + L[6*i+4]*D[6*i+5] + D[6*i+4]*L[6*i+5] for i=0:2),
          sum(L[6*i+5]*D[6*i+5] + L[6*i+3]*D[6*i+6] + D[6*i+3]*L[6*i+6] for i=0:2),
          sum(L[6*i+4]*D[6*i+6] + D[6*i+4]*L[6*i+6] for i=0:2),
          sum(L[6*i+5]*D[6*i+6] + D[6*i+5]*L[6*i+6] for i=0:2),
          sum(L[6*i+6]*D[6*i+6] for i=0:2)
         ] .* 2
    # define the degree 2 part in terms of the step size s in the quadric difference
    d2 = [sum(D[6*i+1]^2 for i=0:2),
          2*sum(D[6*i+1]*D[6*i+2] for i=0:2),
          sum(D[6*i+2]^2 for i=0:2)+2*sum(D[6*i+1]*D[6*i+3] for i=0:2),
          2*sum(D[6*i+2]*D[6*i+3] for i=0:2),
          sum(D[6*i+3]^2 for i=0:2),
          2*sum(D[6*i+1]*D[6*i+4] for i=0:2),
          2*sum(D[6*i+2]*D[6*i+4] + D[6*i+1]*D[6*i+5] for i=0:2),
          2*sum(D[6*i+2]*D[6*i+5] + D[6*i+3]*D[6*i+4] for i=0:2),
          2*sum(D[6*i+3]*D[6*i+5] for i=0:2),
          sum(D[6*i+4]^2 for i=0:2)+2*sum(D[6*i+1]*D[6*i+6] for i=0:2),
          2*sum(D[6*i+2]*D[6*i+6] + D[6*i+4]*D[6*i+5] for i=0:2),
          sum(D[6*i+5]^2 for i=0:2)+2*sum(D[6*i+3]*D[6*i+6] for i=0:2),
          2*sum(D[6*i+4]*D[6*i+6] for i=0:2),
          2*sum(D[6*i+5]*D[6*i+6] for i=0:2),
          sum(D[6*i+6]^2 for i=0:2)
         ]
    # calculate the coefficients of the quartic interpolation
    return [d0'*d0, 2*d0'*d1, d1'*d1+2*d0'*d2, 2*d1'*d2, d2'*d2]
end

# function that tests whether the quartic polynomial is correctly interpolated
function test_interpoloation()
    # test the polynomial with coefficients [0,1,2,3,4]
    let a = [0,1,2,3,4]
        p = Polynomial(a)
        let x = [-1,-0.1,0,0.1,1]
            @test LowRankSOS.interpolate_quartic_Vandermonde(x,p.(x)) ≈ a
        end
        let x = LowRankSOS.Chebx
            @test LowRankSOS.interpolate_quartic_Chebyshev(p.(x)) ≈ a
        end
    end
    # test a numerically challenging polynomial 
    let a = [0.0,1.0,2.0e-3,-3.0e3,4.0e8]
        p = Polynomial(a)
        let x = [-1,-1/3,0,1/3,1]
            @test LowRankSOS.interpolate_quartic_Vandermonde(x,p.(x)) ≈ a
        end
        let x = LowRankSOS.Chebx
            @test LowRankSOS.interpolate_quartic_Chebyshev(p.(x)) ≈ a
        end
    end

    #=================================================
     test the interpolation on the Veronese surface
    ==================================================#
    # linear forms on the Veronese surface correspond to quadrics in x,y,z
    # use the monomial multi-indices to create the multiplication table
    m1 = [[2,0,0],[1,1,0],[0,2,0],[1,0,1],[0,1,1],[0,0,2]]
    m2 = [[4,0,0],[3,1,0],[2,2,0],[1,3,0],[0,4,0],
          [3,0,1],[2,1,1],[1,2,1],[0,3,1],[2,0,2],
          [1,1,2],[0,2,2],[1,0,3],[0,1,3],[0,0,4]]
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
    # test at the forms (x²,x⋅y,y²)
    let L = [1,0,0,0,0,0,
             0,1,0,0,0,0,
             0,0,1,0,0,0],
        # set the target quadric x⁴+x²y²+y⁴+z⁴
        T = [1,0,1,0,1,
             0,0,0,0,0,
             0,0,0,0,1]
        # set the line search direction (0,0,z²)
        let D = [0,0,0,0,0,0,
                 0,0,0,0,0,0,
                 0,0,0,0,0,1]
            # so the difference will be 2s⋅y²z²+(s²-1)⋅z⁴
            # which gives a quartic line search ϕ(s):=(s²-1)²+4s²=s⁴+2s²+1
            P = [1,0,2,0,1]
            # define the objective function restricted to the line
            func_obj = (s) -> norm(LowRankSOS.get_sos(L+s*D,R2)-T,2)^2
            # test the line search quartic interpolation
            @test customized_interp(L,T,D) == P
            @test LowRankSOS.interpolate_quartic_Chebyshev(func_obj.(LowRankSOS.Chebx)) ≈ P
        end
        # set the line search direction (x⋅z,z²,y⋅z)
        let D = [0,0,0,1,0,0,
                 0,0,0,0,0,1,
                 0,0,0,0,1,0]
            # so the difference will be 2s⋅(x³⋅z+x⋅y⋅z²+y³⋅z)+s²(x²z²+z⁴+y²z²)-z⁴
            # which gives a quartic line search ϕ(s):=(s²-1)²+2s⁴+12s²=3s⁴+10s²+1
            P = [1,0,10,0,3]
            # define the objective function restricted to the line
            func_obj = (s) -> norm(LowRankSOS.get_sos(L+s*D,R2)-T,2)^2
            # test the line search quartic interpolation
            @test customized_interp(L,T,D) == P
            @test LowRankSOS.interpolate_quartic_Chebyshev(func_obj.(LowRankSOS.Chebx)) ≈ P
        end
    end
    # test at the forms (x²-y²,2x⋅y,z²)
    let L = [1,0,-1,0,0,0,
             0,2,0,0,0,0,
             0,0,0,0,0,1],
        # set the target quadric x⁴+x²y²+y⁴+z⁴+x⋅y⋅z²
        T = [1,0,1,0,1,
             0,0,0,0,0,
             1,0,0,0,1]
        # set the line search direction (0,z²,x⋅y)
        let D = [0,0,0,0,0,0,
                 0,0,0,0,0,1,
                 0,1,0,0,0,0]
            # so the difference will be (6s-1)x⋅y⋅z²+s²⋅z⁴+(s²+1)x²y²
            # which gives a quartic line search ϕ(s):=s⁴+(s²+1)²+(6s-1)²+1=2s⁴+38s²-12s+2
            P = [2,-12,38,0,2]
            # define the objective function restricted to the line
            func_obj = (s) -> norm(LowRankSOS.get_sos(L+s*D,R2)-T,2)^2
            # test the line search quartic interpolation
            @test customized_interp(L,T,D) == P
            @test LowRankSOS.interpolate_quartic_Chebyshev(func_obj.(LowRankSOS.Chebx)) ≈ P
        end
        # set the line search direction (x²+y²,0,x⋅y)
        let D = [1,0,1,0,0,0,
                 0,0,0,0,0,0,
                 0,1,0,0,0,0]
            # so the difference will be x²y²-x⋅y⋅z²+2s(x⁴-y⁴+x⋅y⋅z²)+s²(x⁴+3x²y²+y⁴)=x⁴(s²+2s)+y⁴(s²-2s)+x²y²(3s²+1)+x⋅y⋅z²(2s-1)
            # which gives a quartic line search ϕ(s):=(s²+2s)²+(s²-2s)²+(3s²+1)²+(2s-1)²=11s⁴+18s²-4s+2
            P = [2,-4,18,0,11]
            # define the objective function restricted to the line
            func_obj = (s) -> norm(LowRankSOS.get_sos(L+s*D,R2)-T,2)^2
            # test the line search quartic interpolation
            @test customized_interp(L,T,D) == P
            @test LowRankSOS.interpolate_quartic_Chebyshev(func_obj.(LowRankSOS.Chebx)) ≈ P
        end
    end
    # test at an instance encountered in the numerical experiment
    let L = [0.15080288549455345, 0.25932071410422025, 0.7067335861950917, 0.03964978360784749, 0.5483396312294921, 0.5435203669040183, 
             0.991109046176464, 0.4077518860467993, 0.6152808877000975, 0.25153110044843147, 1.177349511981686, 0.4192028045616766, 
             0.07008888816712402, 0.29779807295550315, 0.07722097287216814, 0.6921770585166565, -0.041995263089795745, 0.1053420227086608],
        T = [1.2736038064908553, 0.935175145146153, 1.5495533899102447, 0.583517068587137, 0.8217286358349715, 
             0.651608581721135, 3.028157163303601, 1.651823355168321, 2.4405322914626555, 1.6651379332446758, 
             1.4713799592356258, 2.830740764309672, 0.3405789460140853, 1.6253903020320846, 0.4919636901291228],
        D = [0.0005731756324679194, -0.04922606283875168, -0.0443173797179776, -0.01077829192892708, 0.05105774749198357, -0.0005979743039316102, 
             0.04017960568975971, -0.037061914075106915, -0.028955733095284864, 0.004316582228353294, 0.04170887666038825, 0.008019852267672331, 
             0.010499107092181792, -0.0027743770013953233, -0.02392563067125419, 0.0013863199704237906, 0.022580304794403122, -0.016765132080064946]
        # the interpolation result is [0.3802290463406508, -0.17758615602361102, 0.13811124900944247, 0.0016784499123561503, 0.00024165092978010173]
        # the current objective is 0.3802290463406511, and the obtained slope along this line is -0.33175730631855616
            # define the objective function restricted to the line
            func_obj = (s) -> norm(LowRankSOS.get_sos(L+s*D,R2)-T,2)^2
            # test the line search quartic interpolation
            @test customized_interp(L,T,D) ≈ LowRankSOS.interpolate_quartic_Chebyshev(func_obj.(LowRankSOS.Chebx))
    end
end

test_interpoloation()

