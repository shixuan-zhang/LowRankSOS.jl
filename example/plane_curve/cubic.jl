include("./plane_curve.jl")

const MAX_RAND_COEFF = 7

# function that conducts experiments of the low-rank SOS method on the cubic curve
function experiment_cubic_curve(
        curve_coeff::Dict{Vector{Int},T} = Dict{Vector{Int},Int}();
        deg_target::Int = 2,
        num_rep::Int = 1,
        num_square::Int = 3,
        val_tol::Float64 = 1.0e-4
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
        for idx in 1:num_rep
            println("\n" * "="^80)
            # choose randomly a target
            tuple_random = rand(num_square*coord_ring.dim1)
            target_sos = get_sos(tuple_random, coord_ring)
            # choose randomly a starting point
            tuple_start = rand(num_square*coord_ring.dim1)
            # solve the problem and check the optimal value
            #vec_sol, val_res = call_NLopt(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1)
            vec_sol, val_res = solve_BFGS_descent(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, print_level=1, val_tol_norm=val_tol)
            vec_sos = get_sos(vec_sol,coord_ring)
            if norm(vec_sos-target_sos) < val_tol
                vec_success[idx] = 1
            else
                # check the optimality conditions
                vec_sos = get_sos(vec_sol, coord_ring)
                mat_Jac = build_Jac_mat(vec_sol, coord_ring)
                vec_grad = 2*mat_Jac'*(vec_sos-target_sos)
                mat_Hess = build_Hess_mat(num_square, vec_sol, target_sos, coord_ring)
                printfmtln("The res = {:<10.4e}, grad norm = {:<10.4e} and the min eigenval of Hessian = {:<10.4e}",
                           norm(vec_sos-target_sos), norm(vec_grad), minimum(eigen(mat_Hess).values))
                # start the adaptive moves along a direct path connecting the quadrics
                println("Re-solve the problem using the direct path method...")
                vec_sol, val_res = move_direct_path(num_square, target_sos, coord_ring, tuple_linear_forms=tuple_start, str_descent_method="BFGS", print_level=2, val_threshold=val_tol)
                vec_sos = get_sos(vec_sol, coord_ring)
                if norm(vec_sos-target_sos) < val_tol
                    vec_success[idx] = 1
                end
            end
        end
        println("\nGlobal optima are found in ", sum(vec_success), " out of ", num_rep, " experiment runs")
    end
end

experiment_cubic_curve(deg_target=20,num_rep=1000)
