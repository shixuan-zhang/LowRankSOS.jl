# types for low-rank sums-of-squares certification

# structure for storing sparse quadratic forms (symmetric matrices)
struct SparseQuadraticForm
    #TODO: complete the struct
end

# structure for quadrics in the coordinate ring
struct CoordinateRing2
    # dimension of the linear forms R₁ as a real vector space
    dim1::Int 
    # dimension of the linear forms R₂ as a real vector space
    dim2::Int 
    # representations of products of linear forms by the quadric basis
    prod::SparseVector{SparseVector{Rational{Int},Int},Int}
end

