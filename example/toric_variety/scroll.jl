include("./toric.jl")

# function that builds the coordinate ring of a rational normal scroll
function build_ring_scroll(
        vec_height::Vector{Int}
    )
    # get the dimension of the scroll
    dim = length(vec_height)
    # define the lattice polytope vertex matrix
    # where the vertices are from a simplex or certain heights built on it
    mat_simplex = vcat(zeros(Int,dim-1)', diagm(ones(Int,dim-1)))
    mat_vertices = vcat(hcat(mat_simplex,zeros(Int,dim)),hcat(mat_simplex,vec_height))
    # get the coordinate ring information
    coord_ring = build_ring_from_polytope(mat_vertices)
    return coord_ring
end

