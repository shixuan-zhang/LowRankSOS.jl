include("./toric.jl")

# function that builds the coordinate ring of a Veronese variety
function build_ring_Veronese(
        dim::Int,
        deg::Int
    )
    # define the lattice polytope vertex matrix
    mat_vertices = vcat(diagm(ones(Int,dim).*deg),zeros(Int,dim)')
    # get the coordinate ring information
    coord_ring = build_ring_from_polytope(mat_vertices)
    return coord_ring
end
