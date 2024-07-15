include("./plane_curve.jl")

# use packages for experiments and result collection
using Statistics, DataFrames, CSV

const MAX_RAND_COEFF = 7

# function that generates a random plane cubic curve
# re-embedded with a given degree
function generate_plane_cubic(
        deg_target::Int
    )
    # check if the target degree is at least 1
    deg_target = max(deg_target,1)
    # generate the coefficients randomly and check smoothness
    curve_coeff::Dict{Vector{Int},T} = Dict{Vector{Int},Int}()
    while true
        # loop over all monomials under degree 3
        for i=0:3,j=0:(3-i)
            curve_coeff[[i,j]] = rand(-MAX_RAND_COEFF:MAX_RAND_COEFF)
        end
        # get the coordinate ring information
        coord_ring = build_ring_from_plane_curve(curve_coeff,deg_target,print_level=0)
        # check smoothness before re-generation
        if coord_ring.dim1 > 0
            return curve_coeff, coord_ring
        end
    end
end

# function that conducts experiments of the low-rank SOS method on the cubic curve
function experiment_cubic_curve(
        curve_coeff::Dict{Vector{Int},T} = Dict{Vector{Int},Int}();
        deg_target::Int = 2,
        num_rep::Int = 1,
        num_square::Int = 3,
        REL_MAX_ITER::Int = 100
    ) where T <: Union{Int,Rational{Int}}
    # randomly specify the curve if the coefficients are not supplied
    if length(curve_coeff) == 0
        # loop over all monomials under degree 3
        for i=0:3,j=0:(3-i)
            curve_coeff[[i,j]] = rand(-MAX_RAND_COEFF:MAX_RAND_COEFF)
        end
    end
    # check if the target degree is at least 2
    deg_target = max(deg_target,2)
    # get the coordinate ring information
    coord_ring = build_ring_from_plane_curve(curve_coeff,deg_target,print_level=0)
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
        # run the direct path algorithm
        move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, str_descent_method="CG")
        # call the external solver for comparison
        call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
    # run a batch of multiple tests
    elseif num_rep > 1
        # print the experiment setup information
        println("Start experiments on certification of degree-", deg_target, " forms on a cubic curve \n\n")
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
            # solve the problem and check the optimal value
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

function batch_experiment_cubic(
        set_deg::Vector{Int},
        num_curve::Int = 5;
        str_file::String = "result_cubic_curve",
        num_rep::Int = 1000
    )
    # prepare the output columns
    NAME = String[]
    SUCC = Int[]
    FAIL = Int[]
    TIME = Float64[]
    DIST = Float64[]
    # start the main tests
    for idx_curve in 1:num_curve
        # randomly generate a plane cubic curve by looping over all monomials under degree 3
        curve_coeff = Dict{Vector{Int},Int}()
        for i=0:3,j=0:(3-i)
            curve_coeff[[i,j]] = rand(-MAX_RAND_COEFF:MAX_RAND_COEFF)
        end
        # construct its string name for output
        str_curve = ""
        for i=0:3,j=0:(3-i)
            if curve_coeff[[i,j]] != 0
                if curve_coeff[[i,j]] > 0 && i+j > 0
                    str_curve *= "+"
                end
                str_curve *= string(curve_coeff[[i,j]])*"x"*string(i)*"y"*string(j)
            end
        end
        # test with different target degrees
        for deg in set_deg
            # create the name tag from the heights
            push!(NAME, str_curve*":"*string(deg))
            # execute the experiment
            num_succ, num_fail, mean_time, max_dist = experiment_cubic_curve(curve_coeff, deg_target=deg, num_rep=num_rep)
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
end
