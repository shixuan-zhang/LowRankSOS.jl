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
    try 
        (val_opt,sol_opt,status) = NLopt.optimize(opt, tuple_linear_forms)
        time_end = time()
        # check the solution summary
        if print_level >= 0
            println(" "^print_level * "The NLopt l-BFGS solver terminates with status: ", status)
            printfmtln("{} The NLopt returns objective value = {:<10.4e} and uses {:<5.2f} seconds (with {:<5d} evaluations).\n", 
                       " "^print_level, val_opt, time_end-time_start, opt.numevals)
        end
        # check whether the solver has converged
        flag_conv = (status == :SUCCESS)
        return sol_opt, val_opt, flag_conv
    catch err
        println(" "^print_level * "The NLopt solver returns error: " * err.msg, "\n")
        return fill(NaN, num_square*coord_ring.dim1), NaN, false
    end
end

# provide methods of calling external solvers (e.g. CSDP) 
# to solve the full-rank formulation for comparison

using JuMP, CSDP


# function that calls CSDP (interior-point) to solve the full-rank certification problem
function call_CSDP(
        vec_target_quadric::Vector{Float64},
        coord_ring::CoordinateRing2;
        val_threshold::Float64 = VAL_TOL,
        print_level::Int = 0
    )
    # alias the target quadric vector
    F = vec_target_quadric
    # define a JuMP model for SDP feasibility modeling
    M = Model(CSDP.Optimizer)
    set_attribute(M, "printlevel", print_level)
    set_attribute(M, "axtol", val_threshold)
    # define a PSD Gram matrix
    G = @variable(M, [1:coord_ring.dim1,1:coord_ring.dim1], PSD, base_name="G")
    # prepare the linear constraints for each coordinate in R₂
    L = AffExpr[]
    for k in 1:coord_ring.dim2
        push!(L, 0)
    end
    for i in 1:coord_ring.dim1
        for j in i:coord_ring.dim1
            # get the symmetric index of the Gram matrix
            m = idx_sym(i,j)
            # set the multiplicative factor for off-diagonals
            c = (i != j) ? 2 : 1
            # loop over the quadratic monomials to fill in the nonzero entries in the column
            for n in findnz(coord_ring.prod[m])[1]
                L[n] += coord_ring.prod[m][n]*G[i,j]*c
            end
        end
    end
    @constraint(M, [k=1:coord_ring.dim2], L[k] == F[k])
    # set the trivial objective for feasibility only
    @objective(M, Min, sum(G[i,i] for i in 1:coord_ring.dim1))
    # print the model for correctness checking
    if print_level > 0
        println(" "^print_level * "The CSDP model information:")
        show(M)
        println()
    end
    # call the solver and measure the total time
    time_start = time()
    try 
        JuMP.optimize!(M)
        time_end = time()
        # check the solution summary
        if print_level >= 0
            println(" "^print_level * "The CSDP interior-point solver terminates with status: ", primal_status(M))
            printfmtln("{} The CSDP solver uses {:<5.2f} seconds.", 
                       " "^print_level, time_end-time_start)
        end
        if is_solved_and_feasible(M)
            G_sol = value.(G)
            r = rank(G_sol, rtol=val_threshold)
            println(" "^print_level * "The solution has rank = ", r)
            sol_opt = vec(cholesky(G_sol).L)
            return sol_opt, 0.0, true, r
        else
            return fill(NaN,coord_ring.dim1,coord_ring.dim1), NaN, false
        end
    catch err
        println(" "^print_level * "The CSDP solver returns error: " * err.msg)
        return fill(NaN,coord_ring.dim1,coord_ring.dim1), NaN, false
    end
end
