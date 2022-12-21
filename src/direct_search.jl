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
        num_max_step::Int = 100
    )
    # get the current objective value
    val_current = func_obj_val(mat_linear_forms)
    # get the slope at the current point
    val_slope = LinearAlgebra.dot(mat_gradient, mat_direction)
    # initialize the iteration info
    idx_iter = 0
    val_step = val_init_step
    # start the main loop
    while idx_iter <= num_max_step
        # calculate the objective difference
        val_obj_temp = func_obj_val(mat_linear_forms + val_step .* mat_direction)
        # check if there is sufficient descent
        if val_obj_temp <= val_current + val_control * val_step * val_slope
            return val_step
        else
            val_step *= val_reduction
        end
        idx_iter += 1
    end
    return val_step
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
    val_slope = LinearAlgebra.dot(func_obj_diff(mat_linear_forms), mat_direction)
    # sample two points for interpolation
    vec_points = [-1.0, 1.0]
    vec_values = [func_obj_val(mat_linear_forms - mat_direction), 
                  func_obj_val(mat_linear_forms + mat_direction)] .- val_current
    vec_slopes = [LinearAlgebra.dot(func_obj_diff(mat_linear_forms - mat_direction), mat_direction), 
                  LinearAlgebra.dot(func_obj_diff(mat_linear_forms + mat_direction), mat_direction)]
    # interpolate the quartic polynomial
    vec_coefficients = interpolate_quartic_polynomial(vec_points, vec_values, vec_slopes)
    # check if the interpolation is accurate
    if vec_coefficients[1] < 0.0 || abs(val_slope - vec_coefficients[4]) > val_tolerance
        println("DEBUG: the coefficients are ", vec_coefficients)
        println("DEBUG: the current slope is ", val_slope)
        println("DEBUG: the interpolation input is ", vec_values, " and ", vec_slopes)
        println("DEBUG: the current function value is ", val_current)
        println("DEBUG: the current function differential is ", func_obj_diff(mat_linear_forms))
        error("The quartic interpolation returns invalid results!")
    end
    # solve for critical points of the stepsize
    vec_critical_step = sort(find_cubic_roots(vec_coefficients .* [4,3,2,1]))
    # compare the function values to determine the stepsize
    if length(vec_critical_step) == 0
        return 0.0
    elseif length(vec_critical_step) == 1
        return vec_critical_step[1]
    end
    val_obj1 = func_obj_val(mat_linear_forms + vec_critical_step[1] .* mat_direction)
    val_obj3 = func_obj_val(mat_linear_forms + vec_critical_step[3] .* mat_direction)
    if val_obj1 <= val_obj3
        return vec_critical_step[1]
    else
        return vec_critical_step[3]
    end
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
            val_stepsize = line_search_backtracking(mat_linear_forms, -mat_direction, func_obj_val, val_init_step=val_stepsize)
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


# function that finds a descent direction that can be pushed to be close to the difference of quadratic forms
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

