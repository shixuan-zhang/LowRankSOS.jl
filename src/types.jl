# data structure types for low-rank sos certification

# structure for quadrics in the coordinate ring
struct CoordinateRing2
    # dimension of the linear forms R₁ as a real vector space
    dim1::Int 
    # dimension of the linear forms R₂ as a real vector space
    dim2::Int 
    # representations of products of linear forms by the quadric basis
    prod::Vector{SparseVector{Rational{Int},Int}}
end

