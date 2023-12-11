include("./toric.jl")

# function that conducts experiments of the low-rank SOS method on the Veronese variety
function experiment_Veronese(
        deg::Int,
        dim::Int;
        num_rep::Int = 1
    )
    # define the lattice polytope vertex matrix
    mat_vertices = diagm(ones(Int,dim).*deg)
    # get the coordinate ring information
    coord_ring = build_ring_from_polytope(mat_vertices)
    # set the number of squares (that satisfies the Barvinok-Pataki bound)
    num_square = ceil(Int, sqrt(2*binomial(deg+dim,deg)))
    # run a single test
    if num_rep == 1
        # choose randomly a target
        tuple_random = rand(num_square*coord_ring.dim1)
        target_sos = get_sos(tuple_random, coord_ring)
        # run the local descent method
        #TODO: improve efficiency
        #solve_gradient_descent(num_square, target_sos, coord_ring, print=true, num_max_iter=5000)
        # call the external solver for comparison
        call_NLopt(num_square, target_sos, coord_ring)
    # run a batch of multiple tests
    elseif num_rep > 1
        # print the experiment setup information
        println("Start experiments on Veronese variety with degree = ", deg, ", and dimension = ", dim, "\n\n")
        # record whether each experiment run achieves global minimum 0
        vec_success = zeros(Int,num_rep)
        for idx in 1:num_rep
            # choose randomly a target
            tuple_random = rand(num_square*coord_ring.dim1)
            target_sos = get_sos(tuple_random, coord_ring)
            # solve the problem and check the optimal value
            _, val_res = call_NLopt(num_square, target_sos, coord_ring, print=true)
            if val_res < LowRankSOS.VAL_TOL
                vec_success[idx] = 1
            end
        end
        println("\n\nGlobal optima are found in ", sum(vec_success), " out of ", num_rep, " experiment runs")
    end
end

# execute the test
experiment_Veronese(2,10,num_rep=1000)
