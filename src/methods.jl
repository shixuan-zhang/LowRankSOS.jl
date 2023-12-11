## Data structure construction and conversion methods for the low-rank sos problem

# function that converts a pair of indices of two vectors into the
# index of upper triangular entries of a symmetric matrix in the column (grevlex) order
# for example: (1,1),(1,2)=(2,1),(2,2),(1,3)=(3,1),(2,3)=(3,2),(3,3),…
function idx_sym(
        idx1::Int,
        idx2::Int
    )
    i, j = min(idx1,idx2), max(idx1,idx2)
    return Int(j*(j-1)/2 + i)
end

# function that builds the Jacobian of the sum-of-square differential 
# at a given tuple of linear forms (in the form of a sparse matrix)
function build_Jac_mat(
        tuple_linear_forms::Vector{T},
        coord_ring::CoordinateRing2
    ) where T <: Real
    # get the dimension of the tuple
    dim_tuple = length(tuple_linear_forms)
    # get the number of squares
    num_square = dim_tuple ÷ coord_ring.dim1
    if num_square * coord_ring.dim1 != dim_tuple
        error("Invalid length of the input linear forms!")
    end
    # prepare the index and value arrays for the sparse differential matrix
    I, J = Int[], Int[]
    V = T[]
    # loop over each square (in the tuple)
    for k in 1:num_square
        # loop over the monomials corresponding to the columns (index j)
        # and the monomials of the linear forms to be multiplied (index i)
        for i in 1:coord_ring.dim1, j in 1:coord_ring.dim1
            # get the symmetric index of the Gram matrix
            m = idx_sym(i,j)
            # loop over the quadratic monomials to fill in the nonzero entries in the column
            for n in findnz(coord_ring.prod[m])[1]
                push!(I, n)
                push!(J, j+(k-1)*coord_ring.dim1)
                push!(V, 2*coord_ring.prod[m][n]*tuple_linear_forms[i+(k-1)*coord_ring.dim1])
            end
        end
    end
    return sparse(I,J,V,coord_ring.dim2,dim_tuple)
end
# define an alias for the above function
const build_diff_map = build_Jac_mat

# function that calculates the sum of squares of the tuple of linear forms
function get_sos(
        tuple_linear_forms::Vector{T},
        coord_ring::CoordinateRing2
    ) where T <: Real
    # get the number of squares
    num_square, dim_rem = divrem(length(tuple_linear_forms), coord_ring.dim1)
    if dim_rem != 0
        error("Invalid length of the input linear forms!")
    end
    # get the Gram matrix of the linear forms
    L = reshape(tuple_linear_forms, coord_ring.dim1, num_square)
    G = L * L'
    # prepare the output quadric vector
    q = zeros(T,coord_ring.dim2)
    # loop over the upper triangular entries to get the quadric (represented in its basis)
    for i = 1:coord_ring.dim1, j = i:coord_ring.dim1
        # check the monomial in the quadratic forms
        m = idx_sym(i,j)
        # loop over the quadratic monomials to fill in the nonzero entries in the column
        for n in findnz(coord_ring.prod[m])[1]
            q[n] += coord_ring.prod[m][n]*G[i,j]*(i==j ? 1 : 2)
        end
    end
    return q
end

# function that pulls back a linear functional on the space of quadrics
# to a bilinear (quadratic) form on the space of linear forms
function build_bl_form(
        vec_quadric::Vector{T},
        coord_ring::CoordinateRing2
    ) where T <: Real
    # initialize a matrix for the return
    M = zeros(T,coord_ring.dim1,coord_ring.dim1)
    # loop over the upper triangular entries to fill in the bilinear form matrix
    for i = 1:coord_ring.dim1, j = i:coord_ring.dim1
        # check the monomial in the quadratic forms
        m = idx_sym(i,j)
        # loop over the quadratic monomials to add to this entry
        for n in findnz(coord_ring.prod[m])[1]
            M[i,j] += coord_ring.prod[m][n] * vec_quadric[n]
        end
    end
    # return the symmetrized matrix for the bilinear form
    return (M+M')/2
end


# function that builds the Hessian matrix of the sos objective function
function build_Hess_mat(
        num_square::Int,
        tuple_linear_forms::Vector{T},
        vec_quadric::Vector{T},
        coord_ring::CoordinateRing2
    ) where T <: Real
    # initialize a matrix for the return
    d = coord_ring.dim1
    M = zeros(num_square*coord_ring.dim1,num_square*coord_ring.dim1)
    # get the bilinear matrix
    B = build_bl_form(vec_quadric,coord_ring)
    # fill in the block diagonal entries
    for k = 1:num_square
        M[(k-1)*d+1:k*d,(k-1)*d+1:k*d] = B
    end
    # get the Jacobian matrix
    J = build_Jac_mat(tuple_linear_forms,coord_ring)
    # return the sum
    return M + J'*J
end







## General mathematical methods

# function that calculates the real roots of a cubic function explicitly
# see (details)[https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula]
function find_cubic_roots(
        vec_coefficients::Vector{Float64};
        val_tol::Float64 = 1.0e-8
    )
    # check the number of coefficients of the cubic equation
    if length(vec_coefficients) != 4 || vec_coefficients[4] == 0
        error("Invalid input for the cubic equation!")
    end
    # alias the coefficient vector and calculate the modified resultants (Δ₁ and Δ₂)
    a = vec_coefficients[4]
    b = vec_coefficients[3]
    c = vec_coefficients[2]
    d = vec_coefficients[1]
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
# i.e., f(x)=a₄x⁴+a₃x³+a₂x²+a₁x, using function values at four given points
function interpolate_quartic_polynomial(
        vec_points::Vector{T},
        vec_values::Vector{T}
    ) where T <: Real
    # check the input lengths
    if  length(vec_points) != 4 ||
        length(vec_values) != 4 
        error("Invalid input for quartic polynomial interpolation!")
    end
    # form the coefficient matrix 
    C = zeros(4,4)
    for i = 1:4
        C[i,:] = vec_points[i].^[1,2,3,4]
    end
    v = vec_values
    # solve for the coefficients
    return C \ v
end

# function that interpolates a univariate quartic polynomial passing through 0 at 0,
# i.e., f(x)=a₄x⁴+a₃x³+a₂x²+a₁x, using function values and derivatives at two given points
function interpolate_quartic_polynomial(
        vec_points::Vector{T},
        vec_values::Vector{T},
        vec_slopes::Vector{T}
    ) where T <: Real
    # check the input lengths
    if  length(vec_points) != 2 ||
        length(vec_values) != 2 ||
        length(vec_slopes) != 2
        error("Invalid input for quartic polynomial interpolation!")
    end
    # form the coefficient matrix with rows 1 and 2 given by function values
    C = zeros(4,4)
    v = zeros(4)
    C[1,:] = vec_points[1].^[1,2,3,4]
    C[2,:] = vec_points[2].^[1,2,3,4]
    v[1:2] = vec_values
    # and rows 3 and 4 given by function derivatives
    C[3,:] = vec_points[1].^[0,1,2,3] .* [1,2,3,4]
    C[4,:] = vec_points[2].^[0,1,2,3] .* [1,2,3,4]
    v[3:4] = vec_slopes
    # solve for the coefficients
    return C \ v
end


