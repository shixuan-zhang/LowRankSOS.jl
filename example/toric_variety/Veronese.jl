include("./toric.jl")

# function that conducts experiments of the low-rank SOS method on the Veronese variety
function experiment_Veronese(
        deg::Int,
        dim::Int;
        num_rep::Int = 1,
        num_square::Int = -1,
        val_tol::Float64 = 1.0e-4
    )
    # define the lattice polytope vertex matrix
    mat_vertices = vcat(diagm(ones(Int,dim).*deg),zeros(Int,dim)')
    # get the coordinate ring information
    coord_ring = build_ring_from_polytope(mat_vertices)
    # set the number of squares (that satisfies the Barvinok-Pataki bound)
    if num_square < 0
        num_square = ceil(Int, sqrt(2*binomial(deg+dim,deg)))
    end
    # run a single test
    if num_rep == 1
        # choose randomly a target
        tuple_random = rand(num_square*coord_ring.dim1)
        target_sos = get_sos(tuple_random, coord_ring)
        # choose randomly a starting point
        tuple_start = rand(num_square*coord_ring.dim1)
        # run the line search method
        solve_gradient_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
        solve_BFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
        solve_lBFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="PolakRibiere")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="HagerZhang")
        solve_CG_push_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
        solve_CG_push_descent(num_square, target_sos, coord_ring, 1, tuple_linear_forms=tuple_start, print_level=1)
        solve_CG_push_descent(num_square, target_sos, coord_ring, 0, tuple_linear_forms=tuple_start, print_level=1)
        # run the direct path algorithm
        move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_descent_method="CG")
        move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_descent_method="lBFGS")
        # call the external solver for comparison
        call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
    # run a batch of multiple tests
    elseif num_rep > 1
        # print the experiment setup information
        println("Start experiments on Veronese variety with degree = ", deg, ", and dimension = ", dim, "\n\n")
        # record whether each experiment run achieves global minimum 0
        vec_success = zeros(Int,num_rep)
        for idx in 1:num_rep
            println("\n" * "="^80)
            # choose randomly a target
            tuple_random = rand(num_square*coord_ring.dim1)
            target_sos = get_sos(tuple_random, coord_ring)
            # choose randomly a starting point
            tuple_start = rand(num_square*coord_ring.dim1)
            # solve the problem and check the optimal value
            #vec_sol, val_res = call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
            vec_sol, val_res = solve_BFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=0, val_tol_norm=val_tol)
            vec_sos = get_sos(vec_sol,coord_ring)
            if norm(vec_sos-target_sos) < val_tol
                vec_success[idx] = 1
            else
                # check the optimality conditions
                vec_sos = get_sos(vec_sol, coord_ring)
                mat_Jac = build_Jac_mat(vec_sol, coord_ring)
                vec_grad = 2*mat_Jac'*(vec_sos-target_sos)
                mat_Hess = build_Hess_mat(num_square, vec_sol, target_sos, coord_ring)
                printfmtln("The res = {:<10.4e}, grad norm = {:<10.4e} and the min eigenval of Hessian = {:<10.4e}",
                           norm(vec_sos-target_sos), norm(vec_grad), minimum(eigen(mat_Hess).values))
                # start the adaptive moves along a direct path connecting the quadrics
                println("Re-solve the problem using the direct path method...")
                vec_sol, val_res = move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, str_descent_method="BFGS", print_level=1, val_threshold=val_tol)
                vec_sos = get_sos(vec_sol, coord_ring)
                if norm(vec_sos-target_sos) < val_tol
                    vec_success[idx] = 1
                end
            end
        end
        println("\nGlobal optima are found in ", sum(vec_success), " out of ", num_rep, " experiment runs")
    end
end

# conduct the experiments
experiment_Veronese(2,10,num_rep=1000)
