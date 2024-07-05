include("./toric.jl")

# use packages for experiments and result collection
using Statistics, DataFrames, CSV

# function that conducts experiments of the low-rank SOS method on a rational normal scroll
function experiment_scroll(
        vec_deg::Vector{Int};
        num_rep::Int = 1,
        REL_MAX_ITER::Int = 100
    )
    # get the dimension of the scroll
    dim = length(vec_deg)
    # define the lattice polytope vertex matrix
    # where the vertices are from a simplex or certain heights built on it
    mat_simplex = vcat(zeros(Int,dim-1)', diagm(ones(Int,dim-1)))
    mat_vertices = vcat(hcat(mat_simplex,zeros(Int,dim)),hcat(mat_simplex,vec_deg))
    # get the coordinate ring information
    coord_ring = build_ring_from_polytope(mat_vertices)
    # set the number of squares (dim+1)
    num_square = dim+1
    # run a single test
    if num_rep == 1
        # choose randomly a target
        tuple_random = rand(num_square*coord_ring.dim1)
        target_sos = get_sos(tuple_random, coord_ring)
        # choose randomly a starting point
        tuple_start = rand(num_square*coord_ring.dim1)
        # run the line search method
        solve_gradient_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_select_step="backtracking")
        solve_gradient_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_select_step="interpolation")
        solve_BFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_select_step="backtracking")
        solve_BFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_select_step="interpolation")
        solve_lBFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_select_step="backtracking")
        solve_lBFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_select_step="interpolation")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="FletcherReeves", str_select_step="backtracking")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="FletcherReeves", str_select_step="interpolation")
        solve_CG_descent(num_square, target_sos, coord_ring, LowRankSOS.NUM_MAX_ITER, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="FletcherReeves", str_select_step="interpolation")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="PolakRibiere", str_select_step="backtracking")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="PolakRibiere", str_select_step="interpolation")
        solve_CG_descent(num_square, target_sos, coord_ring, LowRankSOS.NUM_MAX_ITER, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="PolakRibiere", str_select_step="interpolation")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="DaiYuan", str_select_step="backtracking")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="DaiYuan", str_select_step="interpolation")
        solve_CG_descent(num_square, target_sos, coord_ring, LowRankSOS.NUM_MAX_ITER, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="DaiYuan", str_select_step="interpolation")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="HagerZhang", str_select_step="backtracking")
        solve_CG_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="HagerZhang", str_select_step="interpolation")
        solve_CG_descent(num_square, target_sos, coord_ring, LowRankSOS.NUM_MAX_ITER, tuple_linear_forms=tuple_start, print_level=1, str_CG_update="HagerZhang", str_select_step="interpolation")
        # run the direct path algorithm
        move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_descent_method="CG")
        move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_descent_method="lBFGS")
        # call the external solver for comparison
        call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
    # run a batch of multiple tests
    elseif num_rep > 1
        # print the experiment setup information
        println("Start experiments on a rational normal scroll with dimension = ", dim, ", and prism heights = ", vec_deg, "\n\n")
        # record whether each experiment run achieves global minimum 0
        vec_success = zeros(Int,num_rep)
        vec_seconds = zeros(num_rep)
        vec_residue = zeros(num_rep)
        for idx in 1:num_rep
            println("\n" * "="^80)
            # choose randomly a target
            tuple_random = rand(num_square*coord_ring.dim1)
            target_sos = get_sos(tuple_random, coord_ring)
            # solve the problem and record the time
            time_start = time()
            vec_sol, val_res, flag_conv = call_NLopt(num_square, target_sos, coord_ring, num_max_eval=REL_MAX_ITER*coord_ring.dim1, print_level=1)
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
                end
            end
        end
        println()
        println("Global optima are found in ", count(x->x>0, vec_success), " out of ", num_rep, " experiment runs.")
        println("The average wall clock time for test runs is ", mean(filter(!isnan, vec_seconds)), " seconds.")
        # return the number of successful runs and the average time for batch experiments
        return count(x->x>0, vec_success), count(x->x<0, vec_success), mean(filter(!isnan, vec_seconds)), maximum(vec_residue)
    end
end

function batch_experiment_scroll(
        set_vec_deg::Vector{Vector{Int}};
        str_file::String = "result_scroll",
        num_rep::Int = 1000
    )
    num_test = length(set_vec_deg)
    # prepare the output columns
    NAME = String[]
    SUCC = Int[]
    FAIL = Int[]
    TIME = Float64[]
    DIST = Float64[]
    # start the main tests
    for idx_test in 1:num_test
        # create the name tag from the heights
        push!(NAME, join(set_vec_deg[idx_test], "-"))
        # execute the experiment
        num_succ, num_fail, mean_time, max_dist = experiment_scroll(set_vec_deg[idx_test], num_rep=num_rep)
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
#experiment_scroll([5,10,15], num_rep=1000)
#experiment_scroll([3,4,5])
batch_experiment_scroll([[5,10],
                         [10,15],
                         [15,20],
                         [30,40],
                         [50,60],
                         [70,80],
                         [5,10,15],
                         [10,20,30],
                         [20,30,40],
                         [5,10,15,20],
                         [10,15,20,25],
                         [15,20,25,30],
                         [5,10,15,20,25]
                        ],
                        str_file = ARGS[1]
                       )
