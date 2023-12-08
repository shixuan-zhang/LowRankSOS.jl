include("./toric.jl")

# function that tests the low-rank SOS method on the Veronese variety
function test_Veronese(
        deg::Int,
        dim::Int
    )
    # define the lattice polytope vertex matrix
    mat_vertices = diagm(ones(Int,dim).*deg)
    # get the coordinate ring information
    coord_ring = build_ring_from_polytope(mat_vertices)
    # set the number of squares (that satisfies the Barvinok-Pataki bound)
    num_square = ceil(Int, sqrt(2*binomial(deg+dim,deg)))
    # choose randomly a target
    tuple_random = rand(num_square*coord_ring.dim1)
    target_sos = get_sos(tuple_random, coord_ring)
    # run the local descent method
    solve_gradient_descent(num_square, target_sos, coord_ring, print=true)
end

# execute the test
test_Veronese(2,10)
