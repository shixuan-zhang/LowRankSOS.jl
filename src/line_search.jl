## Local information-based descent methods
# including gradient and quasi-Newton methods

## step size selection (line search) methods

# function that uses the backtracking heuristic to find a step size
# that satisfies Armijo's (sufficient decrease) condition
function select_step_backtracking(
        tuple_linear_forms::Vector{Float64},
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        vec_grad::Vector{Float64} = Float64[],
        vec_dir::Vector{Float64}  = -vec_grad,
        val_control::Float64      = 0.001,
        val_reduction::Float64    = 0.5,
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
        vec_grad = 2*transpose(mat_diff)*(vec_sos-vec_target_quadric)
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
                println("DEBUG: step size given by backtracking is too small!")
                println("DEBUG: the backtracking history is displayed below.\n", str_step)
            end
            return val_step
        end
        idx_iter += 1
    end
end

# function that uses a quartic interpolation based on finite differences to find a step size
# which satisfies the curvature condition (but not necessarily the decrease condition unless with backtracking)
function select_step_interpolation(
        tuple_linear_forms::Vector{Float64},
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        vec_grad::Vector{Float64} = Float64[],
        vec_dir::Vector{Float64}  = -vec_grad,
        val_threshold::Float64    = 1.0e-4,
        print::Bool               = false
    )
    # get the current sos vector
    vec_sos = get_sos(tuple_linear_forms, coord_ring)
    # get the current objective value
    val_obj = norm(vec_sos-vec_target_quadric,2)^2
    # calculate the gradient if not supplied
    if length(vec_grad) != length(tuple_linear_forms)
        mat_diff = build_diff_map(tuple_linear_forms, coord_ring)
        vec_grad = 2*transpose(mat_diff)*(vec_sos-vec_target_quadric)
        vec_dir = -vec_grad
    end
    # get the slope at the current point (rescaled to the interval [-1,1])
    val_slope = vec_grad' * vec_dir
    # set the objective value evaluation function
    func_obj = (s) -> norm(get_sos(tuple_linear_forms+s*vec_dir,coord_ring)-vec_target_quadric,2)^2
    # do the Chebyshev interpolation on the prescribed interval
    vec_value = func_obj.(Chebx)
    # interpolate the quartic polynomial
    vec_coeff = interpolate_quartic_Chebyshev(vec_value)
    # check if the interpolation is valid
    if vec_coeff[1] < -VAL_TOL
        error("ERROR: the interpolation has a negative constant term and is thus invalid!")
    end
    # check if the interpolation has a significant quartic term
    if vec_coeff[end] < -VAL_TOL
        error("ERROR: the quartic interpolation returns negative quartic term!")
    elseif vec_coeff[end] < val_threshold * (1.0+sum(abs.(vec_coeff[1:3])))
        # limit the stepsize so that a quadratic approximation is valid
        val_max_step = min(abs(vec_coeff[2]/vec_coeff[5])^(1/4), abs(vec_coeff[3]/vec_coeff[5])^(1/2)) / 2
        # determine the stepsize by the quadratic and linear terms
        val_approx_step = -vec_coeff[2] / (2*vec_coeff[3])
        val_step_output = min(val_approx_step, val_max_step)
        # check if the stepsize gives a decrease in the objective value
        if print
            if func_obj(val_step_output) > val_obj + VAL_TOL
                println("DEBUG: use quadratic approximation due to insiginificant quartic term in the line search!")
                println("DEBUG: the interpolation coefficients are ", vec_coeff)
                println("DEBUG: the current function slope is ", val_slope)
                println("DEBUG: the current function value is ", val_obj)
                println("DEBUG: the current linear forms is ", tuple_linear_forms)
                println("DEBUG: the current target quadric is ", vec_target_quadric)
                println("DEBUG: the current search direction is ", vec_dir)
            end
        end
        return val_step_output 
    end
    # solve for critical points of the stepsize based on the quartic interpolation
    vec_critical_step = sort(find_cubic_roots(vec_coeff[2:5] .* [1,2,3,4]))
    # initialize the outputs
    val_step_output = 0.0
    # compare the function values to determine the stepsize
    if length(vec_critical_step) == 0
        return 0.0
    elseif length(vec_critical_step) == 1
        val_step_output = vec_critical_step[1]
    else
        val_obj1 = func_obj(vec_critical_step[1])
        val_obj3 = func_obj(vec_critical_step[3])
        if val_obj1 <= val_obj3
            val_step_output = vec_critical_step[1]
        else
            val_step_output = vec_critical_step[3]
        end
    end
    # check if the step leads to a smaller objective value
    if print
        if func_obj(val_step_output) > val_obj + VAL_TOL
            println("DEBUG: the interpolation coefficients are ", vec_coeff)
            println("DEBUG: the current function slope is ", val_slope)
            println("DEBUG: the current function value is ", val_obj)
            println("DEBUG: the current linear forms is ", tuple_linear_forms)
            println("DEBUG: the current target quadric is ", vec_target_quadric)
            println("DEBUG: the current search direction is ", vec_dir)
        end
    end
    return val_step_output
end


## line search-based descent methods

# function that implements a gradient descent method for the low-rank certification
function solve_gradient_descent(
        num_square::Int,
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        tuple_linear_forms::Vector{Float64} = Float64[],
        num_max_iter::Int = NUM_MAX_ITER,
        val_threshold::Float64 = VAL_TOL,
        print::Bool = false,
        str_select_step::String = "interpolation"
    )
    if print
        println("\n" * "="^80)
    end
    # generate a starting point randomly if not supplied
    if length(tuple_linear_forms) != num_square*coord_ring.dim1
        if print
            println("Start the gradient descent method with a randomly picked point!")
        end
        tuple_linear_forms = rand(num_square*coord_ring.dim1)
    end
    # initialize the iteration info
    idx_iter = 0
    flag_converge = false
    time_start = time()
    val_obj = Inf
    # calculate the initial sos
    vec_init_sos = get_sos(tuple_linear_forms,coord_ring)
    # get gradient vector
    vec_init_grad = 2*transpose(build_diff_map(tuple_linear_forms,coord_ring))*(vec_init_sos-vec_target_quadric)
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
        vec_grad = 2*transpose(build_diff_map(tuple_linear_forms,coord_ring))*(vec_sos-vec_target_quadric)
        if any(isnan.(vec_grad))
            error("ERROR: the gradient contains NaN!")
        end
        # check if the gradient is sufficiently small
        if norm(vec_grad) < val_term 
            flag_converge = true
            break
        end
        # select the stepsize based on the given step selection method
        val_step = 1.0
        vec_dir = -vec_grad
        if str_select_step == "backtracking"
            val_step = select_step_backtracking(tuple_linear_forms,
                                                vec_target_quadric, 
                                                coord_ring, 
                                                vec_grad=vec_grad,
                                                vec_dir =vec_dir,
                                                val_init_step=val_step,
                                                print=print)
        elseif str_select_step == "interpolation"
            val_step = select_step_interpolation(tuple_linear_forms,
                                                 vec_target_quadric, 
                                                 coord_ring, 
                                                 vec_grad=vec_grad,
                                                 vec_dir =vec_dir,
                                                 print=print)
        else
            error("ERROR: unsupported step selection method!")
        end
        # update the current linear forms and the objective
        val_obj = norm(vec_sos-vec_target_quadric,2)^2
        tuple_linear_forms += val_step .* vec_dir
        # print the algorithm progress
        if print
            printfmtln("  Iter {:<4d}: obj = {:<10.6e}, step = {:<10.4e}, grad norm = {:<10.4e}", 
                       idx_iter, val_obj , val_step, norm(vec_grad))
        end
        idx_iter += 1
    end
    # print the number of iterations and total time
    if print
        if flag_converge
            println("The gradient descent method with ", str_select_step, " step selection has converged within ", idx_iter, " iterations!")
        else
            println("The gradient descent method with ", str_select_step, " step selection has exceeded the maximum iterations!")
        end
        println("The gradient descent method with ", str_select_step, " step selection uses ", time() - time_start, " seconds.")
    end
    return tuple_linear_forms, val_obj
end

## Quasi-Newton methods

# function that implements a BFGS descent method for the low-rank certification
function solve_BFGS_descent(
        num_square::Int,
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        tuple_linear_forms::Vector{Float64} = Float64[],
        num_max_iter::Int = NUM_MAX_ITER,
        val_threshold::Float64 = VAL_TOL,
        print::Bool = false,
        str_select_step::String = "interpolation"
    )
    if print
        println("\n" * "="^80)
    end
    # generate a starting point randomly if not supplied
    if length(tuple_linear_forms) != num_square*coord_ring.dim1
        if print
            println("Start the BFGS method with a randomly picked point!")
        end
        tuple_linear_forms = rand(num_square*coord_ring.dim1)
    end
    # initialize the iteration info
    idx_iter = 1
    flag_converge = false
    time_start = time()
    val_obj = Inf
    # calculate the initial sos
    vec_init_sos = get_sos(tuple_linear_forms,coord_ring)
    # get gradient vector
    vec_init_grad = 2*transpose(build_diff_map(tuple_linear_forms,coord_ring))*(vec_init_sos-vec_target_quadric)
    # use the initial norms as scaling factors
    norm_init = norm(tuple_linear_forms)
    norm_init_grad = norm(vec_init_grad)
    val_rescale = min(1.0, (norm_init / norm_init_grad))
    # set the termination threshold based on the initial norm
    val_term = max(val_threshold * sqrt(num_square * coord_ring.dim1), val_threshold * norm_init)
    # initialize the iteration information of visited points and their gradients
    vec_point_old = tuple_linear_forms
    vec_grad_old = vec_init_grad
    vec_point_new = vec_point_old
    vec_grad_new = vec_grad_old
    mat_inv_approx = diagm(ones(num_square*coord_ring.dim1) ./ norm_init_grad)
    val_step_default = 1.0
    # start the main loop
    while idx_iter <= num_max_iter
        # calculate the descent direction by the approximate inverse Hessian matrix
        vec_dir = -mat_inv_approx*vec_grad_old
        # select the stepsize based on the given step selection method
        val_step = val_step_default
        if str_select_step == "backtracking"
            val_step = select_step_backtracking(vec_point_old,
                                                vec_target_quadric, 
                                                coord_ring, 
                                                vec_grad=vec_grad_old,
                                                vec_dir =vec_dir,
                                                val_init_step=val_step,
                                                print=print)
        elseif str_select_step == "interpolation"
            val_step = select_step_interpolation(vec_point_old,
                                                 vec_target_quadric, 
                                                 coord_ring, 
                                                 vec_grad=vec_grad_old,
                                                 vec_dir =vec_dir,
                                                 print=print)
        else
            error("ERROR: unsupported step selection method!")
        end
        # get the new point and its gradient
        vec_point_new = vec_point_old + val_step * vec_dir
        vec_sos_new = get_sos(vec_point_new,coord_ring)
        vec_grad_new = 2*transpose(build_diff_map(vec_point_new,coord_ring))*(vec_sos_new-vec_target_quadric)
        if any(isnan.(vec_grad_new))
            error("ERROR: the gradient contains NaN!")
        end
        # print the algorithm progress
        val_obj = norm(vec_sos_new-vec_target_quadric,2)^2
        if print
            printfmtln("  Iter {:<4d}: obj = {:<10.6e}, step = {:<10.4e}, grad norm = {:<10.4e}", 
                       idx_iter, val_obj, val_step, norm(vec_grad_new))
        end
        # check if the gradient is sufficiently small
        if norm(vec_grad_new) < val_term 
            flag_converge = true
            break
        end
        # otherwise update the point, gradient, and the Hessian inverse approximation
        let H = mat_inv_approx
            s = vec_point_new - vec_point_old
            y = vec_grad_new - vec_grad_old
            ρ = 1.0 / (y'*s)
            mat_inv_approx = (I-ρ*s*y')*H*(I-ρ*y*s') + ρ*s*s'
            # check if the curvature condition is satisfied 1/ρ > 0 
            # (if it is close to zero then stagnancy may occur)
            if abs(y'*s) < VAL_TOL * max(norm(y),norm(s))
                println("DEBUG: the curvature condition is not satisfied with value ", y'*s)
            end
        end
        vec_point_old = vec_point_new
        vec_grad_old = vec_grad_new
        idx_iter += 1
    end
    # print the number of iterations and total time
    if print
        if flag_converge
            println("The BFGS method with ", str_select_step, " step selection has converged within ", idx_iter, " iterations!")
        else
            println("The BFGS method with ", str_select_step, " step selection has exceeded the maximum iterations!")
        end
        println("The BFGS method with ", str_select_step, " step selection uses ", time() - time_start, " seconds.")
    end
    return vec_point_new, val_obj
end

# function that finds a descent direction through limited memory of previous iterations
# the implementation is the ``two-loop L-BFGS method'' 
# (Algorithm 7.4 in Numerical Optimization, Nocedal and Wright 2006, pp.178)
function find_lBFGS_direction(
        vec_grad::Vector{Float64},
        vec_update_point::Vector{Vector{Float64}},
        vec_update_grad::Vector{Vector{Float64}}
    )
    # check the size of the update histories
    n = min(length(vec_update_point), length(vec_update_grad))
    if length(vec_update_point) != length(vec_update_grad)
        println("DEBUG: mismatch in the sizes of iteration histories!")
    end
    if n <= 0
        return -vec_grad
    end
    # initialize the temporary gradient vector
    q = vec_grad
    α = zeros(n)
    ρ = zeros(n)
    # start the first for-loop
    for i in n:-1:1
        ρ[i] = inv(vec_update_grad[i]' * vec_update_point[i])
        α[i] = vec_update_point[i]'*q * ρ[i]
        q -= α[i].*vec_update_grad[i]
    end
    # conduct an initial approximation of the direction
    r = (vec_update_point[n]' * vec_update_grad[n]) /
        (vec_update_grad[n]' * vec_update_grad[n]) .* q
    # start the second for-loop
    for i in 1:n
        β = ρ[i] * (vec_update_grad[i]' * r)
        r += vec_update_point[i] .* (α[i]-β)
    end
    return -r
end


# function that implements a limited-memory BFGS method for the low-rank certification
function solve_lBFGS_descent(
        num_square::Int,
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2,
        num_update_size::Int = NUM_MEM_SIZE;
        tuple_linear_forms::Vector{Float64} = Float64[],
        num_max_iter::Int = NUM_MAX_ITER,
        val_threshold::Float64 = VAL_TOL,
        print::Bool = false,
        str_select_step::String = "interpolation"
    )
    if print
        println("\n" * "="^80)
    end
    # generate a starting point randomly if not supplied
    if length(tuple_linear_forms) != num_square*coord_ring.dim1
        if print
            println("Start the l-BFGS method with a randomly picked point!")
        end
        tuple_linear_forms = rand(num_square*coord_ring.dim1)
    end
    # initialize the iteration info
    idx_iter = 1
    flag_converge = false
    time_start = time()
    val_obj = Inf
    # calculate the initial sos
    vec_init_sos = get_sos(tuple_linear_forms,coord_ring)
    # get gradient vector
    vec_init_grad = 2*transpose(build_diff_map(tuple_linear_forms,coord_ring))*(vec_init_sos-vec_target_quadric)
    # use the initial norms as scaling factors
    norm_init = norm(tuple_linear_forms)
    norm_init_grad = norm(vec_init_grad)
    val_rescale = min(1.0, (norm_init / norm_init_grad))
    # set the termination threshold based on the initial norm
    val_term = max(val_threshold * sqrt(num_square * coord_ring.dim1), val_threshold * norm_init)
    # set the default step size
    val_step_default = 1.0
    # initialize the iteration history for limited memory approximation of descent direction
    vec_update_point = Vector{Float64}[]
    vec_update_grad = Vector{Float64}[]
    vec_point_old = tuple_linear_forms
    vec_grad_old = vec_init_grad
    vec_point_new = vec_point_old
    vec_grad_new = vec_grad_old
    # start the main loop
    while idx_iter <= num_max_iter
        # find the descent direction based on the limited-memory update history
        vec_dir = find_lBFGS_direction(vec_grad_old, vec_update_point, vec_update_grad)
        # select the stepsize based on the given step selection method
        val_step = val_step_default
        if str_select_step == "backtracking"
            val_step = select_step_backtracking(vec_point_old,
                                                vec_target_quadric, 
                                                coord_ring, 
                                                vec_grad=vec_grad_old,
                                                vec_dir =vec_dir,
                                                val_init_step=val_step,
                                                print=false)
            if val_step < VAL_TOL
                println("DEBUG: BFGS step size too small; use gradient direction instead!")
                vec_dir = -vec_grad_old
                val_step = select_step_backtracking(vec_point_old,
                                                    vec_target_quadric, 
                                                    coord_ring, 
                                                    vec_grad=vec_grad_old,
                                                    vec_dir =vec_dir,
                                                    val_init_step=val_step_default,
                                                    print=print)
            end
        elseif str_select_step == "interpolation"
            val_step = select_step_interpolation(vec_point_old,
                                                 vec_target_quadric, 
                                                 coord_ring, 
                                                 vec_grad=vec_grad_old,
                                                 vec_dir =vec_dir,
                                                 print=print)
        else
            error("ERROR: unsupported step selection method!")
        end
        # update the current point
        vec_point_new = vec_point_old + val_step .* vec_dir
        # get the current sos
        vec_sos_new = get_sos(vec_point_new,coord_ring)
        # get gradient vector
        vec_grad_new = 2*transpose(build_diff_map(vec_point_new,coord_ring))*(vec_sos_new-vec_target_quadric)
        if any(isnan.(vec_grad_new))
            error("ERROR: the gradient contains NaN!")
        end
        # print the algorithm progress
        val_obj = norm(vec_sos_new-vec_target_quadric,2)^2
        if print
            printfmtln("  Iter {:<4d}: obj = {:<10.6e}, step = {:<10.4e}, grad norm = {:<10.4e}", 
                       idx_iter, val_obj, val_step, norm(vec_grad_new))
        end
        # check if the gradient is sufficiently small
        if norm(vec_grad_new) < val_term 
            flag_converge = true
            break
        end
        # save the update history
        push!(vec_update_point, vec_point_new-vec_point_old)
        push!(vec_update_grad, vec_grad_new-vec_grad_old)
        vec_point_old = vec_point_new
        vec_grad_old = vec_grad_new
        if length(vec_update_point) > num_update_size
            popfirst!(vec_update_point)
        end
        if length(vec_update_grad) > num_update_size
            popfirst!(vec_update_grad)
        end
        idx_iter += 1
    end
    # print the number of iterations and total time
    if print
        if flag_converge
            println("The l-BFGS method with ", str_select_step, " step selection has converged within ", idx_iter, " iterations!")
        else
            println("The l-BFGS method with ", str_select_step, " step selection has exceeded the maximum iterations!")
        end
        println("The l-BFGS method with ", str_select_step, " step selection uses ", time() - time_start, " seconds.")
    end
    return vec_point_new, val_obj
end




## Conjugate Gradient methods

# function that implements a conjugate gradient method that allows difference choices of update formulas
# more details can be found in Hager and Zhang (2005)
function solve_CG_descent(
        num_square::Int,
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2,
        num_restart_step::Int = num_square;
        tuple_linear_forms::Vector{Float64} = Float64[],
        num_max_iter::Int = NUM_MAX_ITER,
        val_threshold::Float64 = VAL_TOL,
        print::Bool = false,
        str_select_step::String = "interpolation",
        str_CG_update::String = "HagerZhang"
    )
    if print
        println("\n" * "="^80)
    end
    # generate a starting point randomly if not supplied
    if length(tuple_linear_forms) != num_square*coord_ring.dim1
        if print
            println("Start the CG method with a randomly picked point!")
        end
        tuple_linear_forms = rand(num_square*coord_ring.dim1)
    end
    # initialize the iteration info
    idx_iter = 1
    idx_cycle = 0
    flag_converge = false
    time_start = time()
    val_obj = Inf
    # calculate the initial sos
    vec_sos = get_sos(tuple_linear_forms,coord_ring)
    # get gradient vector
    mat_Jac = build_diff_map(tuple_linear_forms,coord_ring)
    vec_grad = 2*mat_Jac'*(vec_sos-vec_target_quadric)
    # use the initial norms as scaling factors
    norm_init = norm(tuple_linear_forms)
    norm_init_grad = norm(vec_grad)
    val_rescale = min(1.0, (norm_init / norm_init_grad))
    # set the termination threshold based on the initial norm
    val_term = max(val_threshold * sqrt(num_square * coord_ring.dim1), val_threshold * norm_init)
    # set the default step size
    val_step_default = 1.0
    # declare the points, directions and gradients used in the iterations
    vec_grad_old = vec_grad
    vec_grad_new = vec_grad
    vec_dir = -vec_grad
    # start the main loop
    while idx_iter <= num_max_iter
        # check if a restart is needed
        if idx_cycle >= num_restart_step
            vec_dir = -vec_grad_old
            idx_cycle = 0
        end
        # select the stepsize based on the given step selection method
        val_step = 1.0
        if str_select_step == "backtracking"
            val_step = select_step_backtracking(tuple_linear_forms,
                                                vec_target_quadric, 
                                                coord_ring, 
                                                vec_grad=vec_grad_old,
                                                vec_dir =vec_dir,
                                                val_init_step=val_step,
                                                print=print)
        elseif str_select_step == "interpolation"
            val_step = select_step_interpolation(tuple_linear_forms,
                                                 vec_target_quadric, 
                                                 coord_ring, 
                                                 vec_grad=vec_grad_old,
                                                 vec_dir =vec_dir,
                                                 print=print)
        else
            error("ERROR: unsupported step selection method!")
        end
        # update the current linear forms and the objective
        tuple_linear_forms += val_step .* vec_dir
        # get the new sos
        vec_sos = get_sos(tuple_linear_forms,coord_ring)
        val_obj = norm(vec_sos-vec_target_quadric,2)^2
        # get the new Jacobian matrix
        mat_Jac = build_diff_map(tuple_linear_forms,coord_ring)
        # get gradient vector
        vec_grad_new = 2*mat_Jac'*(vec_sos-vec_target_quadric)
        if any(isnan.(vec_grad_new))
            error("ERROR: the gradient contains NaN!")
        end
        # print the algorithm progress
        if print
            printfmtln("  Iter {:<4d}: obj = {:<10.6e}, step = {:<10.4e}, grad norm = {:<10.4e}", 
                       idx_iter, val_obj , val_step, norm(vec_grad_new))
        end
        # check if the gradient is sufficiently small
        if norm(vec_grad_new) < val_term 
            flag_converge = true
            break
        end
        # find the new search direction based on the input CG update parameter
        val_CG_update = 0.0
        if str_CG_update == "FletcherReeves"
            val_CG_update = (vec_grad_new'*vec_grad_new)/(vec_grad_old'*vec_grad_old)
        elseif str_CG_update == "PolakRibiere"
            val_CG_update = vec_grad_new'*(vec_grad_new-vec_grad_old)/(vec_grad_old'*vec_grad_old)
        elseif str_CG_update == "DaiYuan"
            val_CG_update = vec_grad_new'*vec_grad_new/(vec_dir'*(vec_grad_new-vec_grad_old))
        elseif str_CG_update == "HagerZhang"
            vec_grad_diff = vec_grad_new-vec_grad_old
            val_prod = vec_dir'*vec_grad_diff
            val_CG_update = (vec_grad_diff-2*vec_dir*(vec_grad_diff'*vec_grad_diff)/val_prod)' * vec_grad_new / val_prod
        else
            error("ERROR: unsupported CG update parameter!")
        end
        # set the new direction
        vec_dir = -vec_grad_new + val_CG_update*vec_dir
        # update the recorded gradient and move to the next iteration
        vec_grad_old = vec_grad_new
        idx_iter += 1
    end
    # print the number of iterations and total time
    if print
        if flag_converge
            println("The CG descent method with ", str_select_step, " step selection and ", str_CG_update, " update has converged within ", idx_iter, " iterations!")
        else
            println("The CG descent method with ", str_select_step, " step selection and ", str_CG_update, " update has exceeded the maximum iterations!")
        end
        println("The CG descent method with ", str_select_step, " step selection and ", str_CG_update, " update uses ", time() - time_start, " seconds.")
    end
    return tuple_linear_forms, val_obj
end

