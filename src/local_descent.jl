## Local information-based descent methods
# including gradient and quasi-Newton methods

## Line search methods

# line search method using backtracking heuristic to reach Armijo-Goldstein condition
function line_search_backtracking(
        tuple_linear_forms::Vector{Float64},
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        vec_grad::Vector{Float64} = Float64[],
        vec_dir::Vector{Float64}  = -vec_grad,
        val_control::Float64      = 0.01,
        val_reduction::Float64    = 0.8,
        val_init_step::Float64    = 1.0,
        val_min_step::Float64     = VAL_TOL,
        print::Bool               = false
    )
    # get the current sos vector
    vec_sos = get_sos(tuple_linear_forms, coord_ring)
    # get the current objective value
    val_obj = norm(vec_sos-vec_target_quadric,2)^2
    # get the gradient if not supplied
    if length(vec_grad) != length(tuple_linear_forms)
        mat_diff = build_diff_map(tuple_linear_forms, coord_ring)
        vec_grad = transpose(mat_diff) * (vec_sos-vec_target_quadric)
        vec_dir = -vec_grad
    end
    # get the slope at the current point
    val_slope = vec_grad' * vec_dir
    # initialize the iteration info
    idx_iter = 0
    val_step = val_init_step
    str_step = ""
    # start the main loop
    while true
        # calculate the objective value at the tentative point
        tuple_temp = tuple_linear_forms + val_step * vec_dir
        val_obj_temp = norm(get_sos(tuple_temp,coord_ring)-vec_target_quadric,2)^2
        # check if there is sufficient descent
        if val_obj_temp <= val_obj + val_control * val_step * val_slope
            return val_step
        else
            val_step *= val_reduction
        end
        if print
            str_step *= format("DEBUG: the tentative step = {:<5.3e}, the candidate obj = {:<10.6e}\n",
                               val_step, val_obj_temp)
        end
        if val_step < val_min_step
            if print
                println("DEBUG: step size given by backtracking line search is too small!")
                println("DEBUG: the backtracking history is displayed below.\n", str_step)
            end
            return val_step
        end
        idx_iter += 1
    end
end

# gradient descent method for low-rank certification
function solve_gradient_method(
        num_square::Int,
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        tuple_linear_forms::Vector{Float64} = Float64[],
        num_max_iter::Int = NUM_MAX_ITER,
        val_threshold::Float64 = VAL_TOL,
        print::Bool = false,
        str_line_search::String = "backtracking"
    )
    # generate a starting point randomly if not supplied
    if length(tuple_linear_forms) != num_square*coord_ring.dim1
        println("Warning: start the algorithm with a randomly picked point!")
        tuple_linear_forms = rand(num_square*coord_ring.dim1)
    end
    # initialize the iteration info
    if print
        println("\n=============================================================================")
    end
    idx_iter = 0
    flag_converge = false
    time_start = time()
    # calculate the initial sos
    vec_init_sos = get_sos(tuple_linear_forms,coord_ring)
    # get gradient vector
    vec_init_grad = transpose(build_diff_map(tuple_linear_forms,coord_ring)) * (vec_init_sos-vec_target_quadric)
    # use the initial norms as scaling factors
    norm_init = norm(tuple_linear_forms)
    norm_init_grad = norm(vec_init_grad)
    val_rescale = min(1.0, (norm_init / norm_init_grad))
    # set the termination threshold based on the initial norm
    val_term = max(val_threshold * sqrt(num_square * coord_ring.dim1), val_threshold * norm_init)
    # start the main loop
    while idx_iter <= num_max_iter
        # get the current sos
        vec_sos = get_sos(tuple_linear_forms,coord_ring)
        # get gradient vector
        vec_grad = transpose(build_diff_map(tuple_linear_forms,coord_ring)) * (vec_sos-vec_target_quadric)
        if any(isnan.(vec_grad))
            error("ERROR: the gradient contains NaN!")
        end
        # check if the gradient is sufficiently small
        if norm(vec_grad) < val_term 
            flag_converge = true
            break
        end
        # select the stepsize based on the given line search method
        val_step = 1.0 / sqrt(norm(vec_grad))
        vec_dir = -vec_grad
        if str_line_search == "backtracking"
            val_step = line_search_backtracking(tuple_linear_forms,
                                                vec_target_quadric, 
                                                coord_ring, 
                                                vec_grad=vec_grad,
                                                vec_dir =vec_dir,
                                                val_init_step=val_step,
                                                print=print)
        else
            error("ERROR: unsupported line search method!")
        end
        # print the algorithm progress
        if print
            printfmtln("  Iter {:<4d}: obj = {:<10.6e}, step = {:<10.4e}, grad norm = {:<10.4e}", 
                       idx_iter, norm(vec_sos-vec_target_quadric,2)^2, val_step, norm(vec_grad))
        end
        # update the current linear forms
        tuple_linear_forms += val_step .* vec_dir
        idx_iter += 1
    end
    # print the number of iterations and total time
    if print
        if flag_converge
            println("The gradient descent method with ", str_line_search, " line search has converged within ", idx_iter, " iterations!")
        else
            println("The gradient descent method with ", str_line_search, " line search has exceeded the maximum iterations!")
        end
        println("The gradient descent method with ", str_line_search, " line search uses ", time() - time_start, " seconds.")
    end
    return tuple_linear_forms
end

