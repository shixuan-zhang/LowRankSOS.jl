# basic methods for low-rank sums-of-squares certification


# information associated with a quadratic (homogeneous) ideal
struct ListDenseQuadrics
    # dimension of the real vector space of linear forms
    dim::Int
    # quadratic form generators of the ideal
    mat_gen::Vector{Matrix{Float64}}
end

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
function construct_quotient_map(list_quad::ListDenseQuadrics; num_digit::Int = NUM_DIG*2)
    # get the dimensions of the linear forms and the quadratic forms (as vectors)
    dim_line = list_quad.dim
    dim_quad = dim_line*(dim_line+1)÷2
    # get the number of ideal generators
    num_gen = length(list_quad.mat_gen)
    # define the matrix with columns being the generators of the ideal
    mat_span = Matrix{Float64}(undef, dim_quad, num_gen)
    for i=1:num_gen
        mat_span[:,i] = convert_sym_to_vec(list_quad.mat_gen[i])
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
function compute_norm_proj(quad_form::Matrix{Float64}, list_quad::ListDenseQuadrics)
    # get the projection matrix
    map_quotient = construct_quotient_map(list_quad)
    # return the norm
    return compute_norm_proj(quad_form, map_quotient)
end







