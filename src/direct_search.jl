# first-order optimization methods for sums-of-squares certification

# function that computes the objective value of the norm squared function
# to the target quadric at a given tuple of linear forms
function compute_obj_val(
        mat_linear_forms::Matrix{Float64}, 
        quad_form::Matrix{Float64}, 
        map_quotient::AbstractMatrix{Float64}
    )
    # get the dimension
    dim = LinearAlgebra.checksquare(quad_form)
    # return the norm of difference
    return LinearAlgebra.norm(convert_vec_to_sym(map_quotient * convert_sym_to_vec(quad_form - mat_linear_forms'*mat_linear_forms), dim=dim))^2
end

# function that computes the gradient of the norm squared function
# to the target quadric at a given tuple of linear forms 
function compute_obj_grad(
        mat_linear_forms::Matrix{Float64}, 
        quad_form::Matrix{Float64}, 
        map_quotient::AbstractMatrix{Float64}
    )
    # get the dimensions
    (rank, dim) = size(mat_linear_forms)
    # get the difference
    mat_diff = convert_vec_to_sym(map_quotient * convert_sym_to_vec(mat_linear_forms'*mat_linear_forms - quad_form), dim=dim)
    # calculate the gradient at each linear form
    mat_grad = 4 .* mat_linear_forms * mat_diff
    return mat_grad
end


# line search method using backtracking heuristic to reach Armijo-Goldstein condition
function line_search_backtracking(
        mat_linear_forms::Matrix{Float64},
        mat_gradient::Matrix{Float64},
        func_obj_val::Function;
        val_control::Float64 = 0.5,
        val_reduction::Float64 = 0.5,
        val_init_step::Float64 = 1.0,
        mat_direction::Matrix{Float64} = -mat_gradient,
        val_min_step::Float64 = VAL_TOL
    )
    # get the current objective value
    val_current = func_obj_val(mat_linear_forms)
    # get the slope at the current point
    val_slope = LinearAlgebra.dot(mat_gradient, mat_direction)
    # initialize the iteration info
    idx_iter = 0
    val_step = val_init_step
    # start the main loop
    while true
        # calculate the objective difference
        val_obj_temp = func_obj_val(mat_linear_forms + val_step .* mat_direction)
        # check if there is sufficient descent
        if val_obj_temp <= val_current + val_control * val_step * val_slope
            return val_step
        else
            val_step *= val_reduction
        end
        if val_step < val_min_step
            println("DEBUG: step size given by backtracking line search is too small!")
            return val_step
        end
        idx_iter += 1
    end
end

# line search method using exact quartic polynomial interpolation and 1-dim global optimization
function line_search_interpolation(
        mat_linear_forms::Matrix{Float64},
        mat_direction::Matrix{Float64},
        func_obj_val::Function,
        func_obj_diff::Function;
        val_sample_step::Float64 = 1.0,
        val_tolerance::Float64 = sqrt(VAL_TOL)
    )
    # get the current objective value
    val_current = func_obj_val(mat_linear_forms)
    # get the current directional derivative (slope)
    mat_grad = func_obj_diff(mat_linear_forms)
    val_slope = LinearAlgebra.dot(mat_grad, mat_direction)
    # sample two points for interpolation
    vec_points = [-val_sample_step, val_sample_step]
    vec_values = [func_obj_val(mat_linear_forms - val_sample_step .* mat_direction), 
                  func_obj_val(mat_linear_forms + val_sample_step .* mat_direction)] .- val_current
    vec_slopes = [LinearAlgebra.dot(func_obj_diff(mat_linear_forms - val_sample_step .* mat_direction), mat_direction), 
                  LinearAlgebra.dot(func_obj_diff(mat_linear_forms + val_sample_step .* mat_direction), mat_direction)]
    # prepare inputs to the interpolation
    while minimum(abs.(vec_values)) < 1.0
        val_sample_step *= 2.0
        vec_points = [-val_sample_step, val_sample_step]
        vec_values = [func_obj_val(mat_linear_forms - val_sample_step .* mat_direction), 
                      func_obj_val(mat_linear_forms + val_sample_step .* mat_direction)] .- val_current
        vec_slopes = [LinearAlgebra.dot(func_obj_diff(mat_linear_forms - val_sample_step .* mat_direction), mat_direction), 
                      LinearAlgebra.dot(func_obj_diff(mat_linear_forms + val_sample_step .* mat_direction), mat_direction)]
        if val_sample_step > 1.0e8
            error("ERROR: cannot find suitable sample step for interpolation!")
        end
    end
    # interpolate the quartic polynomial
    vec_coefficients = interpolate_quartic_polynomial(vec_points, vec_values, vec_slopes)
    # check if the interpolation is accurate
    if vec_coefficients[1] < 0.0 || abs(val_slope - vec_coefficients[4]) > val_tolerance
        println("DEBUG: the coefficients are ", vec_coefficients)
        println("DEBUG: the current slope is ", val_slope)
        println("DEBUG: the values and slopes at sample steps ", val_sample_step, 
                " are ", vec_values, " and ", vec_slopes)
        println("DEBUG: the current function value is ", val_current)
        println("DEBUG: the current function differential is ", func_obj_diff(mat_linear_forms))
        error("The quartic interpolation returns invalid results!")
    end
    # solve for critical points of the stepsize
    vec_critical_step = sort(find_cubic_roots(vec_coefficients .* [4,3,2,1]))
    # initialize the outputs
    val_step_output = 1.0
    # compare the function values to determine the stepsize
    if length(vec_critical_step) == 0
        return 0.0
    elseif length(vec_critical_step) == 1
        val_step_output = vec_critical_step[1]
    else
        val_obj1 = func_obj_val(mat_linear_forms + vec_critical_step[1] .* mat_direction)
        val_obj3 = func_obj_val(mat_linear_forms + vec_critical_step[3] .* mat_direction)
        if val_obj1 <= val_obj3
            val_step_output = vec_critical_step[1]
        else
            val_step_output = vec_critical_step[3]
        end
    end
    # check if the new objective value is smaller
    val_obj_next = func_obj_val(mat_linear_forms + val_step_output .* mat_direction)
    if val_obj_next > val_current
        println("DEBUG: the coefficients are ", vec_coefficients)
        println("DEBUG: the current slope is ", val_slope)
        println("DEBUG: the values and slopes at sample steps ", val_sample_step, 
                " are ", vec_values, " and ", vec_slopes)
        println("DEBUG: the current function value is ", val_current)
        println("DEBUG: the next function value will be ", val_obj_next)
        # return the result from backtracking method instead
        println("The interpolation method fails to give a descent stepsize; use backtracking instead...")
        return line_search_backtracking(mat_linear_forms,
                                        mat_grad,
                                        func_obj_val,
                                        val_init_step = max(1.0,val_step_output)
                                       )
    end
    return val_step_output
end


# gradient descent method for low-rank certification
function solve_gradient_method(
        num_square::Int,
        quad_form::Matrix{Float64},
        map_quotient::AbstractMatrix{Float64};
        mat_linear_forms::Matrix{Float64} = fill(0.0, (0,0)),
        val_threshold::Float64 = VAL_TOL,
        lev_print::Int = 0,
        num_max_iter::Int = NUM_MAX_ITER,
        str_line_search::String = "none"
    )
    # get the dimension
    dim = LinearAlgebra.checksquare(quad_form)
    # generate a starting point randomly if not supplied
    if size(mat_linear_forms) != (num_square, dim)
        mat_linear_forms = randn(num_square, dim)
        println("Warning: start the algorithm with a randomly picked point!")
    end
    # initialize the iteration info
    if lev_print >= 0
        println("\n=============================================================================")
    end
    idx_iter = 0
    flag_converge = false
    time_start = time()
    norm_init = LinearAlgebra.norm(mat_linear_forms)
    norm_init_grad = LinearAlgebra.norm(compute_obj_grad(mat_linear_forms, quad_form, map_quotient))
    val_rescale = min(1.0, (norm_init / norm_init_grad))
    val_term = max(val_threshold * sqrt(num_square * dim), val_threshold * norm_init)
    func_obj_val = (mat_temp)->compute_obj_val(mat_temp, quad_form, map_quotient)
    func_obj_diff = (mat_temp)->compute_obj_grad(mat_temp, quad_form, map_quotient)
    # start the main loop
    while idx_iter <= num_max_iter
        # get the gradient
        mat_grad = compute_obj_grad(mat_linear_forms, quad_form, map_quotient)
        if any(isnan.(mat_grad))
            error("ERROR: the gradient contains NaN!")
        end
        # check if the gradient is sufficiently small
        if LinearAlgebra.norm(mat_grad) < val_term 
            flag_converge = true
            break
        end
        # select the stepsize based on the given line search method
        val_stepsize = LinearAlgebra.norm(mat_grad)
        mat_direction = -mat_grad ./ val_stepsize
        if str_line_search == "backtracking"
            func_obj_val = (mat_temp)->compute_obj_val(mat_temp, quad_form, map_quotient)
            val_stepsize = line_search_backtracking(mat_linear_forms, mat_grad, func_obj_val, val_init_step=val_stepsize)
        elseif str_line_search == "interpolation"
            val_stepsize = line_search_interpolation(mat_linear_forms, mat_direction, func_obj_val, func_obj_diff)
        end
        # print the algorithm progress
        if lev_print >= 1
            println("  Iteration ", idx_iter, ": objective value = ", func_obj_val(mat_linear_forms), ", new step size = ", val_stepsize)
        end
        # update the current linear forms
        mat_linear_forms += val_stepsize .* mat_direction
        idx_iter += 1
    end
    # print the number of iterations and total time
    if lev_print >= 0
        if flag_converge
            println("The gradient descent method with ", str_line_search, " line search has converged within ", idx_iter, " iterations!")
        else
            println("The gradient descent method with ", str_line_search, " line search has exceeded the maximum iterations!")
        end
        println("The gradient descent method with ", str_line_search, " line search uses ", time() - time_start, " seconds.")
    end
    return mat_linear_forms
end


# limited-memory quasi-second-order method for low-rank certification
function solve_limited_memory_method(
        num_square::Int,
        quad_form::Matrix{Float64},
        map_quotient::AbstractMatrix{Float64},
        num_update_size::Int = NUM_MEM_SIZE;
        mat_linear_forms::Matrix{Float64} = fill(0.0, (0,0)),
        val_threshold::Float64 = VAL_TOL,
        lev_print::Int = 0,
        num_max_iter::Int = NUM_MAX_ITER,
        str_line_search::String = "none"
    )
    # get the dimension
    dim = LinearAlgebra.checksquare(quad_form)
    # generate a starting point randomly if not supplied
    if size(mat_linear_forms) != (num_square, dim)
        mat_linear_forms = randn(num_square, dim)
        println("Warning: start the algorithm with a randomly picked point!")
    end
    # initialize the iteration info
    if lev_print >= 0
        println("\n=============================================================================")
    end
    idx_iter = 0
    flag_converge = false
    time_start = time()
    mat_init_grad = compute_obj_grad(mat_linear_forms, quad_form, map_quotient)
    norm_init = LinearAlgebra.norm(mat_linear_forms)
    norm_init_grad = LinearAlgebra.norm(mat_init_grad)
    val_rescale = min(1.0, (norm_init / norm_init_grad))
    val_term = max(val_threshold * sqrt(num_square * dim), val_threshold * norm_init)
    func_obj_val = (mat_temp)->compute_obj_val(mat_temp, quad_form, map_quotient)
    func_obj_diff = (mat_temp)->compute_obj_grad(mat_temp, quad_form, map_quotient)
    # initialize the iteration history for limited memory approximation of descent direction
    vec_updates_point = Vector{Float64}[]
    vec_updates_gradient = Vector{Float64}[]
    vec_grad_old = vec(mat_init_grad)
    vec_point_old = vec(mat_linear_forms)
    mat_grad = mat_init_grad
    # start the main loop
    while idx_iter <= num_max_iter
        # find the descent direction using the limited memory of history updates
        #println("DEBUG: the history of point updates is ", vec_updates_point)
        #println("DEBUG: the history of gradient updates is ", vec_updates_gradient)
        vec_descent = find_descent_direction_limited_memory(vec(mat_grad), vec_updates_point, vec_updates_gradient)
        #println("DEBUG: the found descent direction is ", vec_descent)
        mat_direction = reshape(vec_descent, (num_square,dim))
        # select the stepsize based on the given line search method
        val_stepsize = 1.0
        if abs(LinearAlgebra.dot(mat_direction,mat_grad)) > VAL_TOL
            val_stepsize = max(1.0, inv(abs(LinearAlgebra.dot(mat_direction,mat_grad))))
        end
        if str_line_search == "backtracking"
            func_obj_val = (mat_temp)->compute_obj_val(mat_temp, quad_form, map_quotient)
            val_stepsize = line_search_backtracking(mat_linear_forms, mat_grad, func_obj_val, val_init_step=val_stepsize, mat_direction=mat_direction)
        elseif str_line_search == "interpolation"
            val_stepsize = line_search_interpolation(mat_linear_forms, mat_direction, func_obj_val, func_obj_diff)
        end
        # print the algorithm progress
        if lev_print >= 1
            println("  Iteration ", idx_iter, ": objective value = ", func_obj_val(mat_linear_forms), ", new step size = ", val_stepsize)
        end
        # update the current linear forms
        mat_linear_forms += val_stepsize .* mat_direction
        vec_point_new = vec(mat_linear_forms)
        # get the gradient
        mat_grad = compute_obj_grad(mat_linear_forms, quad_form, map_quotient)
        if any(isnan.(mat_grad))
            println("DEBUG: the linear forms are ", mat_linear_forms)
            println("DEBUG: the direction matrix is ", mat_direction)
            println("DEBUG: the update history for points is ", vec_updates_point)
            println("DEBUG: the update history for gradients is ", vec_updates_gradient)
            error("ERROR: the gradient contains NaN!")
        end
        vec_grad_new = vec(mat_grad)
        # check if the gradient is sufficiently small
        if LinearAlgebra.norm(mat_grad) < val_term 
            flag_converge = true
            break
        end
        # save the update history
        push!(vec_updates_point, vec_point_new-vec_point_old)
        push!(vec_updates_gradient, vec_grad_new-vec_grad_old)
        vec_point_old = vec_point_new
        vec_grad_old = vec_grad_new
        if length(vec_updates_point) > num_update_size
            popfirst!(vec_updates_point)
        end
        if length(vec_updates_gradient) > num_update_size
            popfirst!(vec_updates_gradient)
        end
        idx_iter += 1
    end
    # print the number of iterations and total time
    if lev_print >= 0
        if flag_converge
            println("The limited memory method with ", str_line_search, " line search has converged within ", idx_iter, " iterations!")
        else
            println("The limited memory method with ", str_line_search, " line search has exceeded the maximum iterations!")
        end
        println("The limited memory method with ", str_line_search, " line search uses ", time() - time_start, " seconds.")
    end
    return mat_linear_forms
end




## Obsolete second-order methods (due to high per-iteration costs)
# function that finds a descent direction that can be pushforwarded to be close to the difference of quadratic forms
function find_push_direction(
        mat_linear_forms::Matrix{Float64}, 
        quad_form::Matrix{Float64}, 
        map_quotient::AbstractMatrix{Float64}
    )
    # get the dimensions
    (rank,dim) = size(mat_linear_forms)
    # form the matrix associated with the map from tuples of linear forms to quadratic forms
    mat_ext = Matrix{Float64}(undef, dim*(dim+1)÷2, rank*dim)
    # form the flattened vector associated with target quadratic form of the difference
    vec_diff = map_quotient * convert_sym_to_vec(quad_form - mat_linear_forms'*mat_linear_forms)
    mat_diff = convert_vec_to_sym(vec_diff, dim=dim)
    for i = 1:rank, j = 1:dim
        vec_temp = zeros(dim)
        vec_temp[j] = 1.0
        mat_ext[:,(i-1)*dim+j] = map_quotient * convert_sym_to_vec(vec_temp * mat_linear_forms[i,:]' + mat_linear_forms[i,:] * vec_temp')
    end
    # obtain the search direction as the normalized solution to the linear system
    vec_direction =  mat_ext \ vec_diff 
    mat_direction = copy(reshape(vec_direction, (dim, rank))')
    # check the pushforward direction for debugging
    mat_push_dir = convert_vec_to_sym(map_quotient * convert_sym_to_vec((mat_direction' * mat_linear_forms + mat_linear_forms' * mat_direction) / 2.0), dim=dim)
    mat_grad = -compute_obj_grad(mat_linear_forms, quad_form, map_quotient)
    mat_push_grad = convert_vec_to_sym(map_quotient * convert_sym_to_vec((mat_grad' * mat_linear_forms + mat_linear_forms' * mat_grad) / 2.0), dim=dim)
    return mat_direction ./ LinearAlgebra.norm(mat_direction)
end


# pushforward direction descent method for low-rank certification
function solve_push_method(
        num_square::Int,
        quad_form::Matrix{Float64},
        map_quotient::AbstractMatrix{Float64};
        mat_linear_forms::Matrix{Float64} = fill(0.0, (0,0)),
        val_stepsize::Float64 = 1.0,
        val_threshold::Float64 = VAL_TOL,
        lev_print::Int = 0,
        num_max_iter::Int = NUM_MAX_ITER
    )
    # get the dimension
    dim = LinearAlgebra.checksquare(quad_form)
    # generate a starting point randomly if not supplied
    if size(mat_linear_forms) != (num_square, dim)
        mat_linear_forms = randn(num_square, dim)
        println("Warning: start the algorithm with a randomly picked point!")
    end
    # initialize the iteration info
    if lev_print >= 0
        println("\n=============================================================================")
    end
    idx_iter = 0
    flag_converge = false
    time_start = time()
    func_obj_val = (mat_temp)->compute_obj_val(mat_temp, quad_form, map_quotient)
    func_obj_diff = (mat_temp)->compute_obj_grad(mat_temp, quad_form, map_quotient)
    # start the main loop
    while idx_iter <= num_max_iter
        # get the projected direction
        mat_direction = find_push_direction(mat_linear_forms, quad_form, map_quotient)
        # select the stepsize based on the given line search method
        val_stepsize = line_search_interpolation(mat_linear_forms, mat_direction, func_obj_val, func_obj_diff)
        # terminate the loop if the stepsize is sufficiently small
        if abs(val_stepsize) < val_threshold
            flag_converge = true
            break
        end
        # print the algorithm progress
        if lev_print >= 1
            println("  Iteration ", idx_iter, ": objective value = ", func_obj_val(mat_linear_forms), ", new step size = ", val_stepsize)
        end
        # update the current linear forms
        mat_linear_forms += val_stepsize .* mat_direction
        idx_iter += 1
    end
    # print the number of iterations and total time
    if lev_print >= 0
        if flag_converge
            println("The pushforward descent method with interpolation line search has converged within ", idx_iter, " iterations!")
        else
            println("The pushforward descent method with interpolation line search has exceeded the maximum iterations!")
        end
        println("The pushforward descent method with interpolation line search uses ", time() - time_start, " seconds.")
    end
    return mat_linear_forms
end

