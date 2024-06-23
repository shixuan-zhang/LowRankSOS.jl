# provide methods of calling external solvers (e.g. NLopt) 
# to solve the low-rank sos certification problem

using NLopt

# function that returns the sos objective value and modifies the gradient input as required by NLopt
# the coordinate ring information and target quadric must be prefixed when supplied to NLopt
function obj_NLopt!(
        point::Vector,
        grad::Vector,
        coord_ring::CoordinateRing2,
        vec_target_quadric::Vector
    )
    # calculate the current sos
    vec_sos = get_sos(point, coord_ring)
    # calculate the objective value
    val_obj = norm(vec_sos-vec_target_quadric,2)^2
    # check if the gradient information is requested
    if length(grad) > 0
        mat_diff = build_diff_map(point, coord_ring)
        grad[:] = 2*transpose(mat_diff)*(vec_sos-vec_target_quadric)
    end
    return val_obj
end



# function that calls NLopt (l-BFGS) to solve the low-rank certification problem
function call_NLopt(
        num_square::Int,
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        tuple_linear_forms::Vector{Float64} = Float64[],
        val_threshold::Float64 = VAL_TOL,
        num_max_time::Int = 3600,
        num_max_eval::Int = 2*NUM_MAX_ITER,
        print_level::Int = 0
    )
    # generate a starting point randomly if not supplied
    if print_level > 0
        println("\n"*"="^80)
    end
    if length(tuple_linear_forms) != num_square*coord_ring.dim1
        if print_level > 0
            println(" "^print_level * "Start the NLopt solver with a randomly picked point!")
        end
        tuple_linear_forms = rand(num_square*coord_ring.dim1)
    end
    # define the NLopt objects
    opt = Opt(:LD_LBFGS, num_square*coord_ring.dim1)
    # set the objective function
    opt.min_objective = (l,g)->obj_NLopt!(l,g,coord_ring,vec_target_quadric)
    # set the forced termination criterion
    opt.maxeval = num_max_eval
    opt.maxtime = num_max_time
    # call the solver and measure the total time
    time_start = time()
    (val_opt,sol_opt,status) = optimize(opt, tuple_linear_forms)
    time_end = time()
    # check the solution summary
    if print_level >= 0
        println(" "^print_level * "The NLopt l-BFGS solver terminates with status: ", status)
        printfmtln("{} The NLopt returns objective value = {:<10.4e} and uses {:<5.2f} seconds (with {:<5d} evaluations).", 
                   " "^print_level, val_opt, time_end-time_start, opt.numevals)
    end
    # check whether the solver has converged
    flag_conv = (status == :SUCCESS)
    return sol_opt, val_opt, flag_conv
end
