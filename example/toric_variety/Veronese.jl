include("./toric.jl")

# use packages for experiments and result collection
using Statistics, DataFrames, CSV

# function that conducts experiments of the low-rank SOS method on the Veronese variety
function experiment_Veronese(
        deg::Int,
        dim::Int;
        num_rep::Int = 1,
        num_square::Int = -1,
        val_tol::Float64 = 1.0e-4,
        REL_MAX_ITER::Int = 100
    )
    # define the lattice polytope vertex matrix
    mat_vertices = vcat(diagm(ones(Int,dim).*deg),zeros(Int,dim)')
    # get the coordinate ring information
    coord_ring = build_ring_from_polytope(mat_vertices)
    # set the number of squares (that satisfies the Barvinok-Pataki bound)
    if num_square < 0
        num_square = ceil(Int, sqrt(2*binomial(deg+dim,deg)))
    end
    # run a single test
    if num_rep == 1
        # choose randomly a target
        tuple_random = rand(num_square*coord_ring.dim1)
        target_sos = get_sos(tuple_random, coord_ring)
        # choose randomly a starting point
        tuple_start = rand(num_square*coord_ring.dim1)
        # run the line search method
        solve_gradient_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
        solve_BFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
        solve_lBFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="PolakRibiere")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="HagerZhang")
        solve_CG_push_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
        solve_CG_push_descent(num_square, target_sos, coord_ring, 1, tuple_linear_forms=tuple_start, print_level=1)
        solve_CG_push_descent(num_square, target_sos, coord_ring, 0, tuple_linear_forms=tuple_start, print_level=1)
        # run the direct path algorithm
        move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_descent_method="CG")
        move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_descent_method="lBFGS")
        # call the external solver for comparison
        call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
    # run a batch of multiple tests
    elseif num_rep > 1
        # print the experiment setup information
        println("Start experiments on Veronese variety with degree = ", deg, ", and dimension = ", dim, "\n\n")
        # check the dimension of the linear forms
        dim_linear = binomial(deg+dim,deg)
        # record whether each experiment run achieves global minimum 0
        vec_success = zeros(Int,num_rep)
        vec_seconds = zeros(num_rep)
        vec_residue = zeros(num_rep)
        for idx in 1:num_rep
            println("\n" * "="^80)
            # choose randomly a target
            tuple_random = rand(num_square*coord_ring.dim1)
            target_sos = get_sos(tuple_random, coord_ring)
            # choose randomly a starting point
            tuple_start = rand(num_square*coord_ring.dim1)
            # solve the problem and record the time
            time_start = time()
            vec_sol, val_res, flag_conv = call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, num_max_eval=coord_ring.dim1*REL_MAX_ITER, print_level=1)
            time_end = time()
            # check the optimal value
            if val_res < LowRankSOS.VAL_TOL * max(1.0, norm(target_sos))
                vec_success[idx] = 1
                vec_seconds[idx] = time_end-time_start
            else
                vec_seconds[idx] = NaN
                if flag_conv
                    vec_success[idx] = -1
                    vec_residue[idx] = val_res / max(1.0, norm(target_sos))
                    # check the optimality conditions
                    vec_sos = get_sos(vec_sol, coord_ring)
                    mat_Jac = build_Jac_mat(vec_sol, coord_ring)
                    vec_grad = 2*mat_Jac'*(vec_sos-target_sos)
                    mat_Hess = build_Hess_mat(num_square, vec_sol, target_sos, coord_ring)
                    printfmtln("Local min encountered with grad norm = {:<10.4e} and the min Hessian eigenval = {:<10.4e}",
                               norm(vec_grad), minimum(eigen(mat_Hess).values))
                    # start the adaptive moves along a direct path connecting the quadrics
                    println("Re-solve the problem using the direct path method...")
                    vec_sol, val_res = move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, str_descent_method="lBFGS-NLopt", print_level=1, val_threshold=LowRankSOS.VAL_TOL*max(1.0,norm(target_sos)))
                    vec_sos = get_sos(vec_sol, coord_ring)
                    if norm(vec_sos-target_sos) < LowRankSOS.VAL_TOL*max(1.0,norm(target_sos))
                        vec_success[idx] = 2
                        vec_residue[idx] = 0.0
                    end
                end
            end
        end
        println()
        println("Global optima are found in ", count(x->x>0, vec_success), " out of ", num_rep, " experiment runs.")
        println("Direct-path method is used in ", count(x->x>1, vec_success), " experiment runs.")
        println("The average wall clock time for test runs is ", mean(filter(!isnan, vec_seconds)), " seconds.")
        # return the number of successful runs and the average time for batch experiments
        return count(x->x>0, vec_success), count(x->x<0, vec_success), mean(filter(!isnan, vec_seconds)), maximum(vec_residue)
    end
end

function batch_experiment_Veronese(
        set_deg_dim::Vector{Tuple{Int,Int}};
        str_file::String = "result_Veronese",
        num_rep::Int = 1000
    )
    num_test = length(set_deg_dim)
    # prepare the output columns
    NAME = String[]
    SUCC = Int[]
    FAIL = Int[]
    TIME = Float64[]
    DIST = Float64[]
    # start the main tests
    for idx_test in 1:num_test
        # create the name tag from the heights
        push!(NAME, join(set_deg_dim[idx_test], "-"))
        # execute the experiment
        num_succ, num_fail, mean_time, max_dist = experiment_Veronese(set_deg_dim[idx_test][1], set_deg_dim[idx_test][2], num_rep=num_rep)
        push!(SUCC, num_succ)
        push!(FAIL, num_fail)
        push!(TIME, mean_time)
        push!(DIST, max_dist)
        # write to the output file
        result = DataFrame(:NAME => NAME, :SUCC => SUCC, :TIME => TIME, :FAIL => FAIL, :DIST => DIST)
        CSV.write(str_file*".csv", result)
        println("\n\n\n")
    end
end

# conduct the experiments
#experiment_Veronese(2,10,num_rep=1000)
batch_experiment_Veronese([(2,4),(2,6),(2,8),(2,10),(3,4),(3,6),(4,4)],
                          str_file = ARGS[1]
                         )
