# methods of moving in the fiber to escape local minima or stationary points

# function that finds a moving direction in the fiber to reduce the rank
function find_fiber_direction(
        mat_linear_forms::Matrix{Float64},
        quad_ideal::QuadraticIdeal
    )
    # get the orthogonal basis of the complement of the given linear forms
    mat_kernel = LinearAlgebra.nullspace(mat_linear_forms)
    # get the dimensions
    (num_square, dim) = size(mat_linear_forms)
    null = size(mat_kernel, 2)
    num_gen = length(quad_ideal.mat_gen)
    # check the dimensions
    if null + num_square != dim
        error("The rank and nullity do not match the dimension!")
    end
    # form the linear equations for solving the combination coefficients
    mat_ext = Matrix{Float64}(undef, dim*null+1, num_gen)
    for i=1:null, j=1:num_gen
        mat_ext[((i-1)*dim+1):i*dim,j] = quad_ideal.mat_gen[j] * mat_kernel[:,i]
    end
    # add the normalization equation to avoid trivial solutions (all zero)
    mat_ext[end,:] = ones(num_gen)
    vec_ext = [zeros(dim*null); 1.0]
    # solve for the linear combination coefficients
    vec_combination = mat_ext \ vec_ext
    mat_direction = sum(vec_combination[i] .* quad_ideal.mat_gen[i] for i=1:num_gen)
    return mat_direction
end

# function that finds the perturbed linear forms by moving in the fiber
function find_fiber_perturbation(
        mat_linear_forms::Matrix{Float64},
        mat_fiber_dir::Matrix{Float64};
        val_threshold::Float64 = VAL_TOL
    )
    # check the dimensions
    (num_square, dim) = size(mat_linear_forms)
    if dim != LinearAlgebra.checksquare(mat_fiber_dir)
        error("The dimensions do not match!")
    end
    # get the range of the Gram matrix associated with the linear forms
    vec_eigenval, mat_eigenvec = LinearAlgebra.eigen(mat_linear_forms' * mat_linear_forms, sortby=-)
    mat_range = mat_eigenvec[:, vec_eigenval .> val_threshold]
    # use generalized spectral decomposition to find the stepsize
    vec_genval, mat_genval = LinearAlgebra.eigen(mat_range' * mat_linear_forms' * mat_linear_forms * mat_range, mat_range' * mat_fiber_dir * mat_range)
    _, idx_step = findmin(abs.(vec_genval))
    mat_Gram = mat_linear_forms' * mat_linear_forms - vec_genval[idx_step] .* mat_fiber_dir
    # ensure that the Gram matrix is positive semidefinite
    vec_eigenval, mat_eigenvec = LinearAlgebra.eigen(mat_Gram)
    idx_psd = vec_eigenval .> val_threshold
    num_zero = num_square - sum(idx_psd)
    if num_zero < 1
        # print debugging information to ensure valid convergence
        #=
        println("DEBUG: the current linear forms are ", mat_linear_forms)
        println("DEBUG: the perturbation direction in the fiber is ", mat_fiber_dir)
        println("DEBUG: the perturbation stepsize is ", -vec_genval[idx_step])
        println("DEBUG: the eigenvalues of the perturbed Gram matrix is ", vec_eigenval)
        =#
        println(" Cannot find linear forms with reduced rank in the fiber!")
        return zeros(0,0)
    end
    return [(mat_eigenvec[:,idx_psd] * LinearAlgebra.diagm(sqrt.(vec_eigenval[idx_psd])))'; zeros(num_zero,dim)]
end

# function that finds an improving direction at linearly dependent linear forms
function find_escape_direction(
        mat_linear_forms::Matrix{Float64},
        quad_form::Matrix{Float64},
        map_quotient::AbstractMatrix{Float64};
        val_threshold::Float64 = VAL_TOL
    )
    # check if the last row is all zero
    if LinearAlgebra.norm(mat_linear_forms[end,:]) > val_threshold
        error("Fail to find an escape direction since the linear forms may be independent!")
    end
    (num_square, dim) = size(mat_linear_forms)
    # get the difference Gram matrix
    mat_diff = convert_vec_to_sym(map_quotient * convert_sym_to_vec(mat_linear_forms'*mat_linear_forms - quad_form), dim=dim)
    # use spectral decomposition to find an escape direction
    vec_eigenval, mat_eigenvec = LinearAlgebra.eigen(mat_diff)
    val_min, idx_min = findmin(vec_eigenval)
    if val_min > -val_threshold
        return zeros(0,0)
    else
        return [zeros(num_square-1, dim); mat_eigenvec[:,idx_min]']
    end
end

# solve the gradient descent method with escapes via moving in the fibers
function solve_gradient_method_with_escapes(
        num_square::Int,
        quad_form::Matrix{Float64},
        map_quotient::AbstractMatrix{Float64},
        quad_ideal::QuadraticIdeal;
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
        # get the gradient and the search direction
        mat_grad = compute_obj_grad(mat_linear_forms, quad_form, map_quotient)
        if any(isnan.(mat_grad))
            error("ERROR: the gradient contains NaN!")
        end
        val_stepsize = LinearAlgebra.norm(mat_grad)
        mat_direction = -mat_grad ./ val_stepsize
        # check if the gradient is sufficiently small
        if LinearAlgebra.norm(mat_grad) < val_term 
            if lev_print >= 1
                println("  Near-zero gradient encountered at the current linear forms ", round.(mat_linear_forms,digits=NUM_DIG))
            end
            # find a perturbation that possibly escapes spurious stationary points
            mat_fiber_dir = find_fiber_direction(mat_linear_forms, quad_ideal)
            # get the new temporary linear forms and an escape direction
            mat_linear_forms_temp = find_fiber_perturbation(mat_linear_forms, mat_fiber_dir)
            if size(mat_direction) != (num_square, dim) ||
                size(mat_linear_forms_temp) != (num_square, dim)
                flag_converge = true
                break
            end
            mat_linear_forms = mat_linear_forms_temp
            mat_direction = find_escape_direction(mat_linear_forms_temp, quad_form, map_quotient)
            val_stepsize = 1.0
            if lev_print >= 0
                println(" Search along an escape direction after perturbation in the fiber!")
            end
        end
        # update the stepsize based on the given line search method
        if str_line_search == "backtracking"
            func_obj_val = (mat_temp)->compute_obj_val(mat_temp, quad_form, map_quotient)
            val_stepsize = line_search_backtracking(mat_linear_forms, -mat_direction, func_obj_val, val_init_step=val_stepsize)
        elseif str_line_search == "interpolation"
            val_stepsize = line_search_interpolation(mat_linear_forms, mat_direction, func_obj_val, func_obj_diff)
        end
        # update the current linear forms
        mat_linear_forms += val_stepsize .* mat_direction
        # print the algorithm progress
        if lev_print >= 1
            println("  Iteration ", idx_iter, ": objective value = ", func_obj_val(mat_linear_forms), ", new step size = ", val_stepsize)
        end
        idx_iter += 1
        if lev_print >= 1
            println("  The updated linear forms are ", round.(mat_linear_forms,digits=NUM_DIG))
        end
    end
    # print the number of iterations and total time
    if lev_print >= 0
        if flag_converge
            println("The gradient method has converged within ", idx_iter, " iterations!")
        else
            println("The gradient method has exceeded the maximum iterations!")
        end
        println("The gradient method uses ", time() - time_start, " seconds.")
    end
    return mat_linear_forms
end

