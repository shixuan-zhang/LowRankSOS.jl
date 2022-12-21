# construct the operator of projection onto the subspace perpendicular 
# to the line connecting the starting point and the target
function construct_proj_excursion(
        mat_init_line::Matrix{Float64}, 
        map_quotient::AbstractMatrix{Float64};
        num_digit = NUM_DIG*2
    )
    # get the dimension
    dim = LinearAlgebra.checksquare(mat_init_line)
    # get the projected difference matrix as a vector
    vec_diff = map_quotient * convert_sym_to_vec(mat_init_line)
    # construct the weight matrix for oblique projection
    vec_weight = convert_sym_to_vec(LinearAlgebra.diagm(ones(dim) .* 1/2)) .+ 1/2
    mat_weight = LinearAlgebra.diagm(vec_weight)
    # calculate the oblique projection matrix 
    val_inv = 1.0 / (vec_diff' * mat_weight * vec_diff)
    mat_aux = round.(vec_diff * val_inv * vec_diff' * mat_weight, digits=num_digit)
    map_quotient_excursion = SparseArrays.sparse(LinearAlgebra.I - mat_aux)
    # FIXME: the interpolation is not valid using this quotient map 
    # map_quotient_excursion = LinearAlgebra.I - vec_diff * vec_diff'
    return map_quotient_excursion * map_quotient
end

# function that computes the objective value of the norm squared function
# to the target quadric at a given tuple of linear forms
# with penalty on the excursion from the initial line
function compute_obj_val_with_pen(
        mat_linear_forms::Matrix{Float64}, 
        quad_form::Matrix{Float64}, 
        map_quotient::AbstractMatrix{Float64},
        val_penalty::Float64,
        map_excursion::AbstractMatrix{Float64}
    )
    # get the dimension
    dim = LinearAlgebra.checksquare(quad_form)
    mat_diff = mat_linear_forms'*mat_linear_forms - quad_form
    # calculate the norm of the difference
    val_objective = LinearAlgebra.norm(convert_vec_to_sym(map_quotient * convert_sym_to_vec(mat_diff), dim=dim))^2
    # calculate the penalty of the excursion
    val_excursion = LinearAlgebra.norm(convert_vec_to_sym(map_excursion * convert_sym_to_vec(mat_diff), dim=dim))^2
    # return the norm of difference
    return val_objective + val_penalty * val_excursion
end

# function that computes the gradient of the norm squared function
# to the target quadric at a given tuple of linear forms 
# with penalty on the excursion from the initial line
function compute_obj_grad_with_pen(
        mat_linear_forms::Matrix{Float64}, 
        quad_form::Matrix{Float64}, 
        map_quotient::AbstractMatrix{Float64},
        val_penalty::Float64,
        map_excursion::AbstractMatrix{Float64}
    )
    # get the dimensions
    (rank, dim) = size(mat_linear_forms)
    # get the difference
    mat_diff = mat_linear_forms'*mat_linear_forms - quad_form
    # calculate the gradients corresponding to the objective and the penalty
    vec_objective_grad = map_quotient * convert_sym_to_vec(mat_diff)
    vec_excursion_grad = map_excursion * convert_sym_to_vec(mat_diff)
    # calculate the gradient at each linear form
    mat_grad = 4 .* mat_linear_forms * convert_vec_to_sym(vec_objective_grad + val_penalty * vec_excursion_grad, dim=dim)
    return mat_grad
end


# penalty formulation for low-rank certification to bypass 
# trapping spurious stationary points, solved using gradient descent
function solve_gradient_method_with_penalty(
        num_square::Int,
        quad_form::Matrix{Float64},
        map_quotient::AbstractMatrix{Float64};
        mat_linear_forms::Matrix{Float64} = fill(0.0, (0,0)),
        val_penalty::Float64 = sqrt(1/VAL_TOL),
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
    map_excursion = construct_proj_excursion(quad_form-mat_linear_forms'*mat_linear_forms, map_quotient)
    val_rescale = min(1.0, (norm_init / norm_init_grad))
    val_term = max(val_threshold * sqrt(num_square * dim), val_threshold * norm_init)
    func_obj_val = (mat_temp)->compute_obj_val_with_pen(mat_temp, quad_form, map_quotient, val_penalty, map_excursion)
    func_obj_diff = (mat_temp)->compute_obj_grad_with_pen(mat_temp, quad_form, map_quotient, val_penalty, map_excursion)
    # start the main loop
    while idx_iter <= num_max_iter
        # get the gradient
        mat_grad = compute_obj_grad_with_pen(mat_linear_forms, quad_form, map_quotient, val_penalty, map_excursion)
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
            println("The penalty gradient descent method with ", str_line_search, " line search has converged within ", idx_iter, " iterations!")
        else
            println("The penalty gradient descent method with ", str_line_search, " line search has exceeded the maximum iterations!")
        end
        println("The penalty gradient descent method with ", str_line_search, " line search uses ", time() - time_start, " seconds.")
    end
    return mat_linear_forms
end


# penalty formulation for low-rank certification to bypass 
# trapping spurious stationary points, solved using pushforward descent
function solve_push_method_with_penalty(
        num_square::Int,
        quad_form::Matrix{Float64},
        map_quotient::AbstractMatrix{Float64};
        mat_linear_forms::Matrix{Float64} = fill(0.0, (0,0)),
        val_stepsize::Float64 = 1.0,
        val_threshold::Float64 = VAL_TOL,
        val_penalty::Float64 = sqrt(1/VAL_TOL),
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
    map_excursion = construct_proj_excursion(quad_form-mat_linear_forms'*mat_linear_forms, map_quotient)
    func_obj_val = (mat_temp)->compute_obj_val_with_pen(mat_temp, quad_form, map_quotient, val_penalty, map_excursion)
    func_obj_diff = (mat_temp)->compute_obj_grad_with_pen(mat_temp, quad_form, map_quotient, val_penalty, map_excursion)
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
            println("The penalty pushforward descent method with interpolation line search has converged within ", idx_iter, " iterations!")
        else
            println("The penalty pushforward descent method with interpolation line search has exceeded the maximum iterations!")
        end
        println("The penalty pushforward descent method with interpolation line search uses ", time() - time_start, " seconds.")
    end
    return mat_linear_forms
end

