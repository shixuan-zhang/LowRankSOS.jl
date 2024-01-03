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
        val_threshold::Float64 = 1.0e-4,
        print::Bool = false,
        num_max_move::Int = 500,
        num_move_increase::Int = 2,
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
    val_dist_total = norm(vec_curr_sos-vec_target_quadric)
    val_move_dist = val_dist_total 
    idx_iter = 0
    time_start = time()
    flag_success = false
    # start the main loop
    while idx_iter < num_max_move
        # calculate the move distance
        val_move_dist = min(val_move_dist, val_dist_total)
        # select the new temporary target on the path
        vec_move_dir = (vec_target_quadric-vec_curr_sos)/val_dist_total
        vec_temp_target = vec_curr_sos + val_move_dist*vec_move_dir
        tuple_temp = tuple_linear_forms
        val_res = Inf
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
        # print the progress
        if print
            printfmtln(" Update {:<4d}: target dist = {:<10.6e}, move dist = {:<10.4e}, res = {:<10.4e}", 
                       idx_iter, val_dist_total, val_move_dist, val_res^(1/2))
        end
        # check if the residual is brought down to zero
        if val_res^(1/2) > val_threshold
            val_move_dist *= val_move_decrease
            # terminate the algorithm if no movement can be made
            if val_move_dist < min(VAL_TOL, val_threshold)
                if print
                    println("The direct path algorithm with ", str_descent_method, " method uses ", time() - time_start, " seconds.")
                end
                break
            end
            ctr_move_suc = 0
        else
            tuple_linear_forms = tuple_temp
            vec_curr_sos = get_sos(tuple_linear_forms,coord_ring)
            ctr_move_suc += 1
        end
        # increase the movement size after successively completing the certifications
        if ctr_move_suc > num_move_increase
            val_move_dist = min(val_move_dist/val_move_decrease, val_dist_total)
            ctr_move_suc = 0
        end
        # update the total distance remaining
        val_dist_total = norm(vec_target_quadric-vec_curr_sos) 
        if val_dist_total < val_threshold
            flag_success = true
            break
        end
        idx_iter += 1
    end
    # print the number of iterations and total time
    if print
        if flag_success
            println("The direct path algorithm with ", str_descent_method, " method has completed the certification successfully!")
        else
            println("The direct path algorithm with ", str_descent_method, " method has exceeded the maximum number of updates!")
        end
        println("The direct path algorithm with ", str_descent_method, " method uses ", time() - time_start, " seconds.")
    end
    return tuple_linear_forms, val_dist_total^2
end
