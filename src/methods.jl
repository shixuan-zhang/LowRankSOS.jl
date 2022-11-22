# basic methods for low-rank sums-of-squares certification

# function that converts a symmetric matrix to a vector
function convert_sym_to_vec(mat::AbstractMatrix{T}) where {T <: Any}
    dim = LinearAlgebra.checksquare(mat)
    vec = Vector{T}(undef, dim*(dim+1)÷2)
    for i=1:dim, j=1:dim
        if i > j
            vec[(i-1)*i÷2+j] = mat[i,j] + mat[j,i]
        elseif i == j
            vec[(i-1)*i÷2+j] = mat[i,i]
        end
    end
    return vec
end

# function to convert a vector to a symmetric matrix by the lower triangular part
function convert_vec_to_sym(vec::AbstractVector{T}; dim::Int = 0) where {T <: Any}
    if dim <= 0
        dim = floor(Int, sqrt(2*length(vec)))
    end
    if dim*(dim+1)÷2 != length(vec)
        error("Unable to convert the vector with an invalid length!")
    end
    mat = Matrix{T}(undef,dim,dim)
    for i=1:dim, j=1:dim
        if i > j
            mat[i,j] = vec[(i-1)*i÷2+j] / 2.0
            mat[j,i] = vec[(i-1)*i÷2+j] / 2.0
        elseif i == j
            mat[i,i] = vec[(i-1)*i÷2+j]
        end
    end
    return mat
end


# function to construct the oblique projection matrix associated with the canonical quotient map
# the inner product is weighted for the consistency with the inner product on the space of linear forms
function construct_quotient_map(quad_ideal::QuadraticIdeal; num_digit::Int = NUM_DIG*2)
    # get the dimensions of the linear forms and the quadratic forms (as vectors)
    dim_line = quad_ideal.dim
    dim_quad = dim_line*(dim_line+1)÷2
    # get the number of ideal generators
    num_gen = length(quad_ideal.mat_gen)
    # define the matrix with columns being the generators of the ideal
    mat_span = Matrix{Float64}(undef, dim_quad, num_gen)
    for i=1:num_gen
        mat_span[:,i] = convert_sym_to_vec(quad_ideal.mat_gen[i])
    end
    # check if the matrix has full column rank
    rank = LinearAlgebra.rank(mat_span)
    if rank < num_gen
        println("The quadratic ideal generators are not linearly independent!")
        mat_basis = LinearAlgebra.qr(mat_span).Q
        mat_span = mat_basis[:,1:rank]
    end
    # construct the weight matrix for oblique projection
    vec_weight = convert_sym_to_vec(LinearAlgebra.diagm(ones(dim_line) .* 1/2)) .+ 1/2
    mat_weight = LinearAlgebra.diagm(vec_weight)
    # calculate the oblique projection matrix 
    mat_inv = LinearAlgebra.inv(mat_span' * mat_weight * mat_span)
    mat_aux = round.(mat_span * mat_inv * mat_span' * mat_weight, digits=num_digit)
    map_quotient = SparseArrays.sparse(LinearAlgebra.I - mat_aux)
    return map_quotient
end

# function that computes the norm of a form in projection subspace of quadratic forms
function compute_norm_proj(quad_form::Matrix{Float64}, map_quotient::AbstractMatrix{Float64})
    # ensure the dimensions match
    dim = LinearAlgebra.checksquare(quad_form)
    if dim*(dim+1)÷2 != LinearAlgebra.checksquare(map_quotient)
        error("Projection failed as the dimensions do not match!")
    end
    # convert to vector and then project
    vec_quad_form = map_quotient * convert_sym_to_vec(quad_form)
    # return its norm
    return LinearAlgebra.norm(convert_vec_to_sym(vec_quad_form, dim=dim))
end

# function that computes the norm of a form in the degree 2 part of the coordinate ring
function compute_norm_proj(quad_form::Matrix{Float64}, quad_ideal::QuadraticIdeal)
    # get the projection matrix
    map_quotient = construct_quotient_map(quad_ideal)
    # return the norm
    return compute_norm_proj(quad_form, map_quotient)
end







# function that calculates the real roots of a cubic function explicitly
# see details here: https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
function find_cubic_roots(
        vec_coefficients::Vector{Float64};
        val_tol::Float64 = 1.0e-8
    )
    # check the number of coefficients of the cubic equation
    if length(vec_coefficients) != 4 || vec_coefficients[1] == 0
        error("Invalid input for the cubic equation!")
    end
    # alias the coefficient vector and calculate the modified resultants (Δ1 and Δ2)
    a = vec_coefficients[1]
    b = vec_coefficients[2]
    c = vec_coefficients[3]
    d = vec_coefficients[4]
    Δ0 = b^2 - 3*a*c
    Δ1 = 2*b^3 - 9*a*b*c + 27*a^2*d
    # check if the resultants are zero which implies there is only one real root
    if max(abs(Δ0), abs(Δ1)) < val_tol
        return [-b/(3*a)]
    end
    # otherwise calculate all the three real roots (which may coincide)
    C = ((Δ1 + sqrt(Complex(Δ1^2 - 4*Δ0^3))) / 2)^(1/3)
    ξ = (-1 + sqrt(Complex(-3))) / 2
    x = [-(1/(3*a))*(b + ξ^k*C + ξ^(-k)*Δ0/C) for k=0:1:2]
    # only return the real roots
    return real.(filter(y->abs(imag(y))<val_tol, x))
end

# function that interpolates a univariate quartic polynomial passing through 0 at 0,
# i.e., f(x)=a₄x⁴+a₃x³+a₂x²+a₁x, using function values and derivatives at two given points
function interpolate_quartic_polynomial(
        vec_points::Vector{Float64},
        vec_values::Vector{Float64},
        vec_slopes::Vector{Float64}
    )
    # check the input lengths
    if  length(vec_points) != 2 ||
        length(vec_values) != 2 ||
        length(vec_slopes) != 2
        error("Invalid input for quartic polynomial interpolation!")
    end
    # form the coefficient matrix with rows 1 and 2 given by function values
    C = zeros(4,4)
    v = zeros(4)
    C[1,:] = vec_points[1].^[4, 3, 2, 1]
    C[2,:] = vec_points[2].^[4, 3, 2, 1]
    v[1:2] = vec_values
    # and rows 3 and 4 given by function derivatives
    C[3,:] = vec_points[1].^[3, 2, 1, 0] .* [4, 3, 2, 1]
    C[4,:] = vec_points[2].^[3, 2, 1, 0] .* [4, 3, 2, 1]
    v[3:4] = vec_slopes
    # solve for the coefficients
    return C \ v
end

