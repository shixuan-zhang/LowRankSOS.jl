include("./plane_curve.jl")

# set the range for random cubic coefficient generation
const MAX_RAND_COEFF = 7

# function that generates a random plane cubic curve
# re-embedded with a given degree
function generate_plane_cubic(
        deg_target::Int
    )
    # check if the target degree is at least 1
    deg_target = max(deg_target,1)
    while true
        # generate the coefficients randomly and check smoothness
        curve_coeff::Dict{Vector{Int},Int} = Dict{Vector{Int},Int}()
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

