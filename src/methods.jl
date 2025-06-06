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
    M = zeros(coord_ring.dim1,coord_ring.dim1)
    # loop over the upper triangular entries to fill in the bilinear form matrix
    for i = 1:coord_ring.dim1, j = i:coord_ring.dim1
        # check the monomial in the quadratic forms
        m = idx_sym(i,j)
        # loop over the quadratic monomials to add to this entry
        for n in findnz(coord_ring.prod[m])[1]
            M[i,j] += coord_ring.prod[m][n] * vec_quadric[n]
            if i != j
                M[j,i] += coord_ring.prod[m][n] * vec_quadric[n]
            end
        end
    end
    # return the symmetrized matrix for the bilinear form
    return M
end


# function that builds the Hessian matrix of the sos objective function
function build_Hess_mat(
        num_square::Int,
        tuple_linear_forms::Vector{T},
        vec_quadric::Vector{S},
        coord_ring::CoordinateRing2
    ) where {T <: Real, S <: Real}
    # initialize a matrix for the return
    d = coord_ring.dim1
    M = zeros(num_square*coord_ring.dim1,num_square*coord_ring.dim1)
    # get the bilinear matrix
    B = build_bl_form(get_sos(tuple_linear_forms,coord_ring)-vec_quadric,coord_ring)
    # fill in the block diagonal entries
    for k = 1:num_square
        M[(k-1)*d+1:k*d,(k-1)*d+1:k*d] = B
    end
    # get the Jacobian matrix
    J = build_Jac_mat(tuple_linear_forms,coord_ring)
    # return the sum (and multiplied by 2 by the def of Hessian)
    return 2*(2*M + J'*J)
end


# function that embeds the tuple into a larger tuple of linear forms
function embed_tuple(
        tuple::Vector{Float64},
        old_num_sq::Int,
        new_num_sq::Int;
        random::Bool = false
    )
    # check the input numbers of squares
    if old_num_sq > new_num_sq
        error("Reducing the number of squares is not supported!")
    end
    # get the dimension of linear forms
    dim, rem = divrem(length(tuple), old_num_sq)
    if rem != 0
        error("Invalid number of squares as input!")
    end
    # embed the tuple by filling zeros
    L = [tuple; zeros((new_num_sq-old_num_sq)*dim)]
    if random
        Q,_ = qr(rand(new_num_sq,new_num_sq))
        M = reshape(L, dim, new_num_sq)
        L = vec(M*Q)
    end
    return L
end

# function that calculates an upper (Barvinok-Pataki) bound on the
# Pythagoras number, which is used as the number of squares
function get_BP_bound(
    coord_ring::CoordinateRing2
    )
    # get the dimension of the quadrics
    dim2 = coord_ring.dim2
    # pick the smallest integer k such that (k+1) choose 2 ≥ dim of quadrics - 1
    k = floor(Int,sqrt(2*dim2 - 1))
    if k*(k+1) < 2*dim2 - 1
        k += 1
    end
    return k
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

# function that interpolates a univariate quartic polynomial, i.e., 
# f(x)=a₄x⁴+a₃x³+a₂x²+a₁x+a₀, using function values at five given points
function interpolate_quartic_Vandermonde(
        vec_points::Vector{T},
        vec_values::Vector{T}
    ) where T <: Real
    # check the input lengths
    if  length(vec_points) != 5 ||
        length(vec_values) != 5 
        error("Invalid input for quartic polynomial interpolation!")
    end
    # form the coefficient matrix 
    C = zeros(5,5)
    for i = 1:5
        C[i,:] = vec_points[i].^[0,1,2,3,4]
    end
    v = vec_values
    # solve for the coefficients
    return C \ v
end

# constants for quartic Chebyshev interpolation nodes on [-1,1]
const Cheb₁ = cos(pi/10)
const Cheb₂ = cos(pi*3/10)
const Chebx = [-Cheb₁, -Cheb₂, 0, Cheb₂, Cheb₁]
const Chebf = [ChebyshevT(diagm(ones(5))[j,:]) for j=1:5]
const ChebF = [Chebf[i](Chebx[j]) for i=1:5, j=1:5]

# function that interpolates a general univariate quartic polynomial i.e., 
# f(x)=a₄x⁴+a₃x³+a₂x²+a₁x+a₀ where x∈[-1,1], using Chebyshev interpolation method
function interpolate_quartic_Chebyshev(
        vec_value::Vector{Float64}
    )
    # check the input length
    if length(vec_value) != 5
        error("Invalid input for quartic polynomial interpolation!")
    end
    # get the Chebyshev coefficient vector
    vec_Chebc = ChebF * vec_value .* 2/5
    vec_Chebc[1] /= 2.0
    # convert the coefficients
    vec_coeff = Polynomial(ChebyshevT(vec_Chebc)).coeffs
    if length(vec_coeff) < 5
        vec_coeff = vcat(vec_coeff,zeros(5-length(vec_coeff)))
    end
    return vec_coeff
end

