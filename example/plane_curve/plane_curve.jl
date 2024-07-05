using Singular
using LinearAlgebra, SparseArrays
using Formatting

include("../../src/LowRankSOS.jl")
using .LowRankSOS

# function that sets up the coordinate ring (degree d and 2d) information
# from a plane curve of degree e ≤ d, where d is the degree of the summands
# the input dictionary has keys of monomial exponents (of x₁ and x₂), 
# which will be homogenized by x₀, and the values are their coefficients
function build_ring_from_plane_curve(
        dict_coeff::Dict{Vector{Int},T},
        deg_target::Int;
        print_level::Int = -1,
        check_smooth::Bool = true
    ) where T <: Union{Int,Rational{Int}}
    # check whether the input has valid lengths
    if maximum(length.(keys(dict_coeff))) > 2
        error("ERROR: the input coefficients must come from a plane curve!")
    end
    # check the total degree of the input polynomial
    deg = maximum(collect(sum.(keys(dict_coeff))))
    if deg > 2*deg_target
        error("ERROR: the degree of target quadrics is smaller than the curve!")
    end
    # declare the polynomial ring in Singular
    R,(x₀,x₁,x₂) = PolynomialRing(QQ,["x0","x1","x2"])
    # define the curve and the ideal
    f = 0
    for (a, c) in dict_coeff
        f += c*x₁^a[1]*x₂^a[2]*x₀^(deg-a[1]-a[2])
    end
    if print_level >= 0
        println(" "^print_level * "The plane curve is defined by ", f)
        println(" "^print_level * "We aim to certify a sum of degree-", deg_target, " forms.")
    end
    I = std(Ideal(R,f))
    # check whether the curve is smooth
    if check_smooth
        if dimension(std(jacobian_ideal(f))) > 0
            error("ERROR: the plane curve is singular!")
        end
    end
    # get the vector space bases of the degree-d part of the coordinate ring
    b1 = kbase(I,deg_target)
    b2 = kbase(I,deg_target*2)
    dim1 = ngens(b1)
    dim2 = ngens(b2)
    if print_level >= 0
        println(" "^print_level * "The dimension of linear forms = ", dim1)
        println(" "^print_level * "The dimension of quadratic forms = ", dim2)
        if print_level > 0
            println(" "^print_level * "The basis for linear forms are ", b1)
            println(" "^print_level * "The basis for quadratic forms are ", b2)
        end
    end
    # create a dictionary to facilitate lookups of the quadric basis
    m2 = Dict(b2[i]=>i for i in 1:dim2)
    # declare the sparse vector for the product table
    prod = sparsevec(Int[],SparseVector{Rational{Int},Int}[],Int(dim1*(dim1+1)/2))
    # loop over the monomials to get the representations of the products
    for i=1:dim1, j=1:dim1
        m = b1[i]*b1[j]
        # get the representation by Singular
        r = reduce(m,I)
        idx = Int[]
        val = Rational{Int}[]
        for k = 1:dim2
            if coeff(r,b2[k]) != 0
                push!(idx,k)
                push!(val,coeff(r,b2[k]))
            end
        end
        prod[LowRankSOS.idx_sym(i,j)] = sparsevec(idx,val,dim2)
    end
    return LowRankSOS.CoordinateRing2(dim1,dim2,prod)
end
