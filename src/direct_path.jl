## High-level algorithms for SOS certification
# where one starts from a point through a direct path 
# to the target quadric by restricting the step lengths

# function that solves a sequence of SOS certification problems
# along a direct path connecting the initial and target quadrics
function move_direct_path(
        num_square::Int,
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        tuple_linear_forms::Vector{Float64} = Float64[],
        val_threshold::Float64 = VAL_TOL,
        print::Bool = false,
        num_move_increase::Int = 10,
        val_move_decrease::Float64 = 0.5,
        val_min_move::Float64 = 1.0e-2,
        str_descent_method::String = "BFGS",
        str_select_step::String = "interpolation"
    )
    if print
        println("\n" * "="^80)
    end
    # generate a starting point randomly if not supplied
    if length(tuple_linear_forms) != num_square*coord_ring.dim1
        if print
            println("Start the direct path method with a randomly picked tuple!")
        end
        tuple_linear_forms = rand(num_square*coord_ring.dim1)
    end
    # get the initial sos and the associated path
    vec_init_sos = get_sos(tuple_linear_forms,coord_ring)
    vec_curr_sos = vec_init_sos
    ctr_move_suc = 0
    val_move_dist = 1.0
    idx_iter = 0
    # start the main loop
    while norm(vec_curr_sos-vec_target_quadric) > val_threshold
        # select the new temporary target on the path
        vec_temp_target = vec_curr_sos + val_move_dist*(vec_target_quadric-vec_curr_sos)
        tuple_temp = tuple_linear_forms
        val_res = 0.0
        # calculate the total distances
        val_dist_total = norm(vec_target_quadric-vec_curr_sos) 
        val_move_dist_abs = val_move_dist*val_dist_total
        # solve the local descent subproblem for certification
        if str_descent_method == "BFGS"
            tuple_temp, val_res = solve_BFGS_descent(num_square, 
                                                     vec_temp_target, 
                                                     coord_ring, 
                                                     tuple_linear_forms=tuple_linear_forms, 
                                                     str_select_step=str_select_step)
        elseif str_descent_method == "lBFGS"
            tuple_temp, val_res = solve_lBFGS_descent(num_square, 
                                                      vec_temp_target, 
                                                      coord_ring, 
                                                      tuple_linear_forms=tuple_linear_forms, 
                                                      str_select_step=str_select_step)
        elseif str_descent_method == "CG"
            tuple_temp, val_res = solve_CG_descent(num_square, 
                                                   vec_temp_target, 
                                                   coord_ring, 
                                                   tuple_linear_forms=tuple_linear_forms, 
                                                   str_select_step=str_select_step)
        else
            error("ERROR: unsupported local descent method for the direct path algorithm!")
        end
        # check if the residual is brought down to zero
        if val_res > val_threshold
            val_move_dist *= val_move_decrease
        else
            tuple_linear_forms = tuple_temp
            vec_curr_sos = get_sos(tuple_linear_forms,coord_ring)
            ctr_move_suc += 1
        end
        # terminate the algorithm if no movement can be made
        if val_move_dist_abs < val_threshold
            if print
                println("The algorithm is termianted because no movement can be made!")
            end
            break
        end
        # increase the movement size after successively completing the certifications
        if ctr_move_suc > num_move_increase
            val_move_dist = min(val_move_dist/val_move_decrease, 1.0)
            ctr_move_suc = 0
        end
        # print the progress
        if print
            printfmtln(" Update {:<4d}: target dist = {:<10.6e}, move dist = {:<10.4e}, res = {:<10.4e}", 
                       idx_iter, val_dist_total, val_move_dist_abs, val_res)
        end
        idx_iter += 1
    end
    return tuple_linear_forms, norm(vec_curr_sos-vec_target_quadric,2)^2
end
