function exec_multiple(
        coord_ring::CoordinateRing2,
        set_num_sq::Vector{Int}
    )
    # record whether each experiment run achieves global minimum 0
    result_success = Dict{Int,Vector{Int}}()
    result_seconds = Dict{Int,Vector{Float64}}()
    result_residue = Dict{Int,Vector{Float64}}()
    for num in set_num_sq
        result_success[num] = zeros(Int,NUM_REPEAT)
        result_seconds[num] = zeros(NUM_REPEAT)
        result_residue[num] = zeros(NUM_REPEAT)
    end
    for idx in 1:NUM_REPEAT
        println("\n" * "="^80)
        # choose randomly a target
        tuple_random = rand(coord_ring.dim1*coord_ring.dim1)
        target_sos = get_sos(tuple_random, coord_ring)
        # find the minimum number of squares
        num_sq_min = minimum(set_num_sq)
        # choose randomly a starting point
        tuple_min_sq = rand(num_sq_min*coord_ring.dim1)
        # loop over different numbers of squares
        for num_square in set_num_sq
            println("Use ", num_square, " squares for the local optimization method...")
            # embed the starting tuple of linear forms
            tuple_start = embed_tuple(tuple_min_sq, num_sq_min, num_square, random=true)
            # solve the problem and record the time
            time_start = time()
            vec_sol, val_res, flag_conv = call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, num_max_eval=coord_ring.dim1*REL_MAX_ITER, print_level=1) 
            time_end = time()
            # check the optimal value
            if val_res < VAL_TOL * max(1.0, norm(target_sos))
                result_success[num_square][idx] = 1
                result_seconds[num_square][idx] = time_end-time_start
            else
                result_seconds[num_square][idx] = NaN
                if flag_conv
                    result_success[num_square][idx] = -1
                    result_residue[num_square][idx] = val_res / max(1.0, norm(target_sos))
                    # check the optimality conditions
                    vec_sos = get_sos(vec_sol, coord_ring)
                    mat_Jac = build_Jac_mat(vec_sol, coord_ring)
                    vec_grad = 2*mat_Jac'*(vec_sos-target_sos)
                    mat_Hess = build_Hess_mat(num_square, vec_sol, target_sos, coord_ring)
                    printfmtln("Stationary point encountered with grad norm = {:<10.4e} and the min Hessian eigenval = {:<10.4e}",
                               norm(vec_grad), minimum(eigen(mat_Hess).values))
                    # start the adaptive moves along a direct path connecting the quadrics
                    println("Re-solve the problem using the direct path method...")
                    vec_sol, val_res = move_direct_path(num_square, target_sos, coord_ring, 
                                                        tuple_linear_forms=tuple_start, 
                                                        str_descent_method="lBFGS-NLopt", 
                                                        print_level=1, 
                                                        val_threshold=VAL_TOL*max(1.0,norm(target_sos)))
                    vec_sos = get_sos(vec_sol, coord_ring)
                    if norm(vec_sos-target_sos) < VAL_TOL*max(1.0,norm(target_sos))
                        result_success[num_square][idx] = 2
                        result_residue[num_square][idx] = 0.0
                    end
                end
            end
        end
    end
    for num in set_num_sq
        println("\nResult summary for ", num, " squares:")
        println("Global optima are found in ", count(x->x>0, result_success[num]), " out of ", NUM_REPEAT, " experiment runs.")
        println("Restricted-path method is used in ", count(x->x>1, result_success[num]), " experiment runs.")
        println("The average wall clock time for test runs is ", mean(filter(!isnan, result_seconds[num])), " seconds.")
    end
    # return the arrays of successful runs and the average time for batch experiments
    SUCC = [count(x->x>0, result_success[num]) for num in set_num_sq]
    FAIL = [count(x->x<0, result_success[num]) for num in set_num_sq]
    TIME = [mean(filter(!isnan, result_seconds[num])) for num in set_num_sq]
    DIST = [maximum(result_residue[num]) for num in set_num_sq]
    return SUCC, FAIL, TIME, DIST
end
