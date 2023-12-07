using Polymake

include("../../src/LowRankSOS.jl")
using .LowRankSOS

# function that sets up the coordinate ring information from vertices of a lattice polytope
function build_ring_from_polytope(
        mat_vertices::Matrix{Int}
    )
    # get the size of the input matrix
    num_vertex, dim_lattice = size(mat_vertices)
    # create the polytope using Polymake
    p = polytope.Polytope(POINTS=hcat(ones(Int,num_vertex),mat_vertices))
    # get the basis of the linear forms (each row is an index for a monomial)
    b1 = p.LATTICE_POINTS_GENERATORS[1][:,2:end]
    dim1, _ = size(b1)
    # get the polytope of the degree-2 part
    q = polytope.minkowski_sum(p,p)
    # get the basis of the quadrics (each row is an index for a monomial)
    b2 = q.LATTICE_POINTS_GENERATORS[1][:,2:end]
    dim2, _ = size(b2)
    # create a dictionary to facilitate lookups of the quadric basis
    m2 = Dict(b2[i,:]=>i for i in 1:dim2)
    # declare the sparse vector for the product table
    prod = sparsevec(Int[],SparseVector{Rational{Int},Int}[],dim1*(dim1+1)/2)
    # loop over the linear monomials to get the product table
    for i in 1:dim1
        for j in i:dim1
            v =  b1[:,i]+b1[:,j]
            n = m2[v]
            prod[get_sym(i,j)] = sparsevec([n],[1],dim2)
        end
    end
    return CoordinateRing2(dim1,dim2,prod)
end
