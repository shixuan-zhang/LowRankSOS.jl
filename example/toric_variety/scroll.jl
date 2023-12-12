include("./toric.jl")

# function that conducts experiments of the low-rank SOS method on a rational normal scroll
function experiment_scroll(
        vec_deg::Vector{Int};
        num_rep::Int = 1
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
    # run a single test
    if num_rep == 1
        # choose randomly a target
        tuple_random = rand(num_square*coord_ring.dim1)
        target_sos = get_sos(tuple_random, coord_ring)
        # run the local descent method
        #TODO: improve efficiency
        #solve_gradient_descent(num_square, target_sos, coord_ring, print=true, num_max_iter=5000)
        # call the external solver for comparison
        call_NLopt(num_square, target_sos, coord_ring, print=true)
    # run a batch of multiple tests
    elseif num_rep > 1
        # print the experiment setup information
        println("Start experiments on a rational normal scroll with dimension = ", dim, ", and polytope heights = ", vec_deg, "\n\n")
        # record whether each experiment run achieves global minimum 0
        vec_success = zeros(Int,num_rep)
        for idx in 1:num_rep
            # choose randomly a target
            tuple_random = rand(num_square*coord_ring.dim1)
            target_sos = get_sos(tuple_random, coord_ring)
            # solve the problem and check the optimal value
            vec_sol, val_res = call_NLopt(num_square, target_sos, coord_ring, print=true)
            if val_res < LowRankSOS.VAL_TOL
                vec_success[idx] = 1
            else
                # check the optimality conditions
                vec_sos = get_sos(vec_sol, coord_ring)
                mat_Jac = build_Jac_mat(vec_sol, coord_ring)
                vec_grad = mat_Jac' * (vec_sos-target_sos)
                mat_Hess = build_Hess_mat(num_square, vec_sol, target_sos, coord_ring)
                printfmtln("The grad norm = {:<10.4e} and the min eigenval of Hessian = {:<10.4e}",
                           norm(vec_grad), minimum(eigen(mat_Hess).values))
            end
        end
        println("\nGlobal optima are found in ", sum(vec_success), " out of ", num_rep, " experiment runs")
    end
end

experiment_scroll(collect(2:8), num_rep=1000)
#experiment_scroll([100,200])
