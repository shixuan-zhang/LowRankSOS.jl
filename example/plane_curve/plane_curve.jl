using LinearAlgebra, SparseArrays
using Formatting

include("../../src/LowRankSOS.jl")
using .LowRankSOS

# function that sets up the coordinate ring (degree d and 2d) information
# from a plane curve of degree e ≤ d, where d is the degree of the target
function build_ring_from_plane_curve(
        dict_coeff::Dict{Vector{Int},T},
        deg_target::Int
    ) where T <: Real
    # TODO: implement this function using Singular.jl?
end
