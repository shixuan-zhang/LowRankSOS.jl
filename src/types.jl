# types for low-rank sums-of-squares certification

# information associated with a quadratic (homogeneous) ideal
struct QuadraticIdeal
    # dimension of the real vector space of linear forms
    dim::Int
    # quadratic form generators of the ideal
    mat_gen::Vector{Matrix{Float64}}
end
