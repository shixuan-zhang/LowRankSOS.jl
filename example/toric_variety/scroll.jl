include("./toric.jl")

# function that tests the low-rank SOS method on a rational normal scroll
function test_scroll(
        vec_deg::Vector{Int}
    )
    # get the dimension of the scroll
    dim = length(vec_deg)
    # define the lattice polytope vertex matrix
    # where the vertices are from a simplex or certain heights built on it
    mat_simplex = vcat(zeros(Int,dim-1)', diagm(ones(Int,dim-1)))
    mat_vertices = vcat(hcat(mat_simplex,zeros(Int,dim)),hcat(mat_simplex,vec_deg))
    # get the coordinate ring information
    coord_ring = build_ring_from_polytope(mat_vertices)
    # set the number of squares (dim+1)
    num_square = dim+1
    # choose randomly a target
    tuple_random = rand(num_square*coord_ring.dim1)
    target_sos = LowRankSOS.get_sos(tuple_random, coord_ring)
    # run the local descent method
    LowRankSOS.solve_gradient_method(num_square, target_sos, coord_ring, print=true, num_max_iter=50)
end

test_scroll([3,4,5,6,7])
