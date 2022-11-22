# optimization models for sums-of-squares certification

# import modules for modeling and solving the low-rank optimization
import JuMP         # for optimization modeling
import Ipopt, NLopt # for solving nonlinear optimization models
import SDPA, CSDP   # for solving semidefinite optimization models

# nonlinear optimization model for low-rank certification
# with auxiliary variables representing the equivalence relations from the ideal
function solve_nonlinear_model(
        num_square::Int,
        quad_form::Matrix{Float64},
        quad_ideal::QuadraticIdeal;
        mat_linear_forms::Matrix{Float64} = zeros(0)
    )
    # check the input dimensions
    dim = quad_ideal.dim
    if size(quad_form) != (dim,dim)
        error("The dimensions of the linear forms do not match!")
    end
    # get the number of quadratic form generators of the ideal
    num_gen = length(quad_ideal.mat_gen)
    # define the optimization model
    model = JuMP.Model()
    var_linear_forms = JuMP.@variable(model, [1:num_square, 1:dim], base_name="l")
    var_combination = JuMP.@variable(model, [1:num_gen], base_name="a")
    expr_Gram_matrix = JuMP.@expression(model, var_linear_forms' * var_linear_forms)
    expr_difference = JuMP.@expression(model, expr_Gram_matrix + sum(var_combination[i] .* quad_ideal.mat_gen[i] for i=1:num_gen) - quad_form)
    JuMP.@NLobjective(model, Min, sum(2*expr_difference[i,j]^2 for i=1:dim,j=1:dim if i<j) + sum(expr_difference[i,i]^2 for i=1:dim))
    # set the starting points if supplied, or otherwise randomly choose one
    if size(mat_linear_forms) != (num_square, dim)
        mat_linear_forms = randn(num_square, dim)
    end
    for i=1:num_square, j=1:dim
        JuMP.set_start_value(var_linear_forms[i,j], mat_linear_forms[i,j])
    end
    for i=1:num_gen
        JuMP.set_start_value(var_combination[i], 0.0)
    end
    # solve the optimization model using an external solver
    JuMP.set_optimizer(model, NLopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "algorithm", :LD_LBFGS)
    JuMP.optimize!(model)
    # check the solution summary
    println("\n=============================================================================")
    println("The nonlinear model terminates with status: ", JuMP.termination_status(model))
    println("The solver run time is ", JuMP.solve_time(model), " seconds.")
    # retrieve the solutions
    mat_linear_forms = fill(NaN, (num_square, dim))
    if JuMP.has_values(model)
        mat_linear_forms = JuMP.value.(var_linear_forms)
    end

    return mat_linear_forms
end


# nonlinear optimization model for low-rank certification
# with auxiliary variables representing the equivalence relations from the ideal
# and restriction to the line connecting the starting point and the target quadric
function solve_restricted_nonlinear_model(
        num_square::Int,
        quad_form::Matrix{Float64},
        quad_ideal::QuadraticIdeal;
        val_tol::Float64 = 1.0e-2,
        mat_linear_forms::Matrix{Float64} = zeros(0)
    )
    # check the input dimensions
    dim = quad_ideal.dim
    if size(quad_form) != (dim,dim)
        error("The dimensions of the linear forms do not match!")
    end
    # get the number of quadratic form generators of the ideal
    num_gen = length(quad_ideal.mat_gen)
    # generate a starting point randomly if not supplied
    if size(mat_linear_forms) != (num_square, dim)
        mat_linear_forms = randn(num_square, dim)
    end
    mat_initial_Gram = mat_linear_forms' * mat_linear_forms
    # define the optimization model
    model = JuMP.Model()
    var_linear_forms = JuMP.@variable(model, [1:num_square, 1:dim], base_name="l")
    var_combination = JuMP.@variable(model, [1:num_gen], base_name="a")
    var_restriction = JuMP.@variable(model, base_name="b")
    expr_Gram_matrix = JuMP.@expression(model, var_linear_forms' * var_linear_forms)
    expr_difference = JuMP.@expression(model, expr_Gram_matrix + sum(var_combination[i] .* quad_ideal.mat_gen[i] for i=1:num_gen) - quad_form)
    expr_excursion = JuMP.@expression(model, expr_difference - var_restriction .* (mat_initial_Gram - quad_form))
    JuMP.@NLconstraint(model, sum(2*expr_excursion[i,j]^2 for i=1:dim,j=1:dim if i<j) + sum(expr_excursion[i,i]^2 for i=1:dim) <= val_tol)
    JuMP.@NLobjective(model, Min, sum(2*expr_difference[i,j]^2 for i=1:dim,j=1:dim if i<j) + sum(expr_difference[i,i]^2 for i=1:dim))
    # set the starting point
    for i=1:num_square, j=1:dim
        JuMP.set_start_value(var_linear_forms[i,j], mat_linear_forms[i,j])
    end
    for i=1:num_gen
        JuMP.set_start_value(var_combination[i], 0.0)
    end
    JuMP.set_start_value(var_restriction, 1.0)
    # solve the optimization model using an external solver
    JuMP.set_optimizer(model, NLopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "algorithm", :LD_MMA)
    JuMP.optimize!(model)
    # check the solution summary
    println("\n=============================================================================")
    println("The restricted nonlinear model terminates with status: ", JuMP.termination_status(model))
    println("The solver run time is ", JuMP.solve_time(model), " seconds.")
    # retrieve the solutions
    mat_linear_forms = fill(NaN, (num_square, dim))
    if JuMP.has_values(model)
        mat_linear_forms = JuMP.value.(var_linear_forms)
        println("The restriction variable value is ", JuMP.value(var_restriction))
    end

    return mat_linear_forms
end


# nonlinear optimization model for low-rank certification
# with auxiliary variables representing the equivalence relations from the ideal
# and penalty on excursion from the line connecting the starting point and the target quadric
function solve_penalized_nonlinear_model(
        num_square::Int,
        quad_form::Matrix{Float64},
        quad_ideal::QuadraticIdeal;
        val_pen::Float64 = 1.0e4,
        mat_linear_forms::Matrix{Float64} = zeros(0)
    )
    # check the input dimensions
    dim = quad_ideal.dim
    if size(quad_form) != (dim,dim)
        error("The dimensions of the linear forms do not match!")
    end
    # get the number of quadratic form generators of the ideal
    num_gen = length(quad_ideal.mat_gen)
    # generate a starting point randomly
    if size(mat_linear_forms) != (num_square, dim)
        mat_linear_forms = randn(num_square, dim)
    end
    mat_initial_Gram = mat_linear_forms' * mat_linear_forms
    # define the optimization model
    model = JuMP.Model()
    var_linear_forms = JuMP.@variable(model, [1:num_square, 1:dim], base_name="l")
    var_combination = JuMP.@variable(model, [1:num_gen], base_name="a")
    var_restriction = JuMP.@variable(model, base_name="b")
    expr_Gram_matrix = JuMP.@expression(model, var_linear_forms' * var_linear_forms)
    expr_difference = JuMP.@expression(model, expr_Gram_matrix + sum(var_combination[i] .* quad_ideal.mat_gen[i] for i=1:num_gen) - quad_form)
    expr_excursion = JuMP.@expression(model, expr_difference - var_restriction .* (mat_initial_Gram - quad_form))
    JuMP.@NLobjective(model, Min, sum(2*(expr_difference[i,j]^2 + expr_excursion[i,j]^2) for i=1:dim,j=1:dim if i<j) + sum(expr_difference[i,i]^2 + expr_excursion[i,i]^2 for i=1:dim))
    # set the starting point
    for i=1:num_square, j=1:dim
        JuMP.set_start_value(var_linear_forms[i,j], mat_linear_forms[i,j])
    end
    for i=1:num_gen
        JuMP.set_start_value(var_combination[i], 0.0)
    end
    JuMP.set_start_value(var_restriction, 1.0)
    # solve the optimization model using an external solver
    JuMP.set_optimizer(model, NLopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "algorithm", :LD_LBFGS)
    JuMP.optimize!(model)
    # check the solution summary
    println("\n=============================================================================")
    println("The penalized nonlinear model terminates with status: ", JuMP.termination_status(model))
    println("The solver run time is ", JuMP.solve_time(model), " seconds.")
    # retrieve the solutions
    mat_linear_forms = fill(NaN, (num_square, dim))
    if JuMP.has_values(model)
        mat_linear_forms = JuMP.value.(var_linear_forms)
        println("The restriction variable value is ", JuMP.value(var_restriction))
    end

    return mat_linear_forms
end

# semidefinite optimization (feasibility) model for (full-rank) certification
function solve_semidefinite_model(
        quad_form::Matrix{Float64},
        quad_ideal::QuadraticIdeal
    )
    # check the input dimensions
    dim = quad_ideal.dim
    if size(quad_form) != (dim,dim)
        error("The dimensions of the linear forms do not match!")
    end
    # get the number of quadratic form generators of the ideal
    num_gen = length(quad_ideal.mat_gen)
    # define the optimization model
    model = JuMP.Model()
    var_Gram_matrix = JuMP.@variable(model, [1:dim, 1:dim], PSD, base_name="G")
    var_combination = JuMP.@variable(model, [1:num_gen], base_name="a")
    JuMP.@constraint(model, var_Gram_matrix + sum(var_combination[i] .* quad_ideal.mat_gen[i] for i in 1:num_gen) .== quad_form)
    JuMP.@objective(model, Min, 0)
    # solve the optimization model using an external solver
    JuMP.set_optimizer(model, CSDP.Optimizer)
    JuMP.set_optimizer_attribute(model, "printlevel", 0)
    JuMP.optimize!(model)
    # check the solution summary
    println("\n=============================================================================")
    println("The semidefinite model terminates with status: ", JuMP.termination_status(model))
    println("The solver run time is ", JuMP.solve_time(model), " seconds.")
    # retrieve the solutions
    mat_Gram = fill(NaN, (dim, dim))
    if JuMP.has_values(model)
        mat_Gram = JuMP.value.(var_Gram_matrix)
    end

    return mat_Gram
end
