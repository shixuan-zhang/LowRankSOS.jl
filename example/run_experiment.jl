include("../src/LowRankSOS.jl")
using .LowRankSOS
# import the constructions of the example varieties
include("toric_variety/Veronese.jl")
include("toric_variety/scroll.jl")
include("plane_curve/cubic.jl")
# import modules for batch test statistics and output
using Statistics, DataFrames, CSV
using Formatting

# set the experiment parameters
const VAL_TOL = 1.0e-8
const NUM_REPEAT = 100
const REL_MAX_ITER = 100
const SCROLL_HEIGHTS = [[5,10],[10,15],[15,20],
                        [30,40],[50,60],[70,80],
                        [5,10,15],[10,20,30],[20,30,40],
                        [5,10,15,20],[10,15,20,25],
                        [15,20,25,30],[5,10,15,20,25]]
const CUBIC_CURVE_DEG = [10,20,30,40,50]
const VERONESE_DIM_DEG = [(4,2),(6,2),(8,2),(10,2),
                          (3,3),(4,3),(5,3),
                          (2,4),(3,4),(2,5)]

const NUM_SQ_ADD = [0,1,2,3]
const NUM_SQ_MULT = [1.0, 1.1, 1.2, 1.5, 2.0]

# include functions that execute single or multiple experiments
include("exec_single.jl")
include("exec_multiple.jl")


# check the command line arguments
num_args = length(ARGS)
## by default we run experiments on all examples
str_example = "all" # or "scroll", "cubic", or "Veronese"
if num_args > 0
    str_example = ARGS[1]
end
## by default we run batch experiments and retrieve the statistics
### when `num_repeat == 1` we only run the experiment with the first parameter in the arrays
num_repeat = NUM_REPEAT
if num_args > 1
    num_repeat = parse(Int, ARGS[2])
end
## set the default output file name
str_output = "result_"*str_example*".csv"
if num_args > 2
    str_output = ARGS[3]
end

# function that conducts experiments on the specified examples
function run_experiments()
    if num_repeat == 1
        if str_example == "scroll" || str_example == "all"
            vec_height = SCROLL_HEIGHTS[1]
            dim = length(vec_height)
            coord_ring = build_ring_scroll(vec_height)
            exec_single(coord_ring, dim+1)
        end
        if str_example == "cubic" || str_example == "all"
            deg_target = CUBIC_CURVE_DEG[1]
            _, coord_ring = generate_plane_cubic(deg_target)
            exec_single(coord_ring, 3)
        end
        if str_example == "Veronese" || str_example == "all"
            dim, deg = VERONESE_DIM_DEG[1]
            coord_ring = build_ring_Veronese(dim, deg)
            num_sq = get_BP_bound(coord_ring)
            exec_single(coord_ring, num_sq)
        end
    else
        # prepare the inputs
        EXAMPLE = String[]
        IDX2PAR = Int[]
        if str_example == "scroll" || str_example == "all"
            append!(EXAMPLE, ["scroll" for _ in 1:length(SCROLL_HEIGHTS)])
            append!(IDX2PAR, collect(1:length(SCROLL_HEIGHTS)))
        end
        if str_example == "cubic" || str_example == "all"
            append!(EXAMPLE, ["cubic" for _ in 1:length(CUBIC_CURVE_DEG)])
            append!(IDX2PAR, collect(1:length(CUBIC_CURVE_DEG)))
        end
        if str_example == "Veronese" || str_example == "all"
            append!(EXAMPLE, ["Veronese" for _ in 1:length(VERONESE_DIM_DEG)])
            append!(IDX2PAR, collect(1:length(VERONESE_DIM_DEG)))
        end
        # prepare the outputs
        NAME = String[]
        RANK = Int[]
        SUCC = Int[]
        FAIL = Int[]
        TIME = Float64[]
        DIST = Float64[]
        SDPT = Float64[]
        SDPR_MIN = Int[]
        SDPR_MED = Int[]
        # start the main experiments
        num_experiment = length(EXAMPLE)
        vec_rank = Int[]
        coord_ring = CoordinateRing2(0,0,SparseVector{Rational{Int},Int}[])
        curve_coeff = Dict{Vector{Int},Int}() # for cubic curves only
        str_curve = ""
        for idx = 1:num_experiment
            if EXAMPLE[idx] == "scroll"
                vec_height = SCROLL_HEIGHTS[IDX2PAR[idx]]
                # create the name tag from the heights
                append!(NAME, ["Scroll:"*join(vec_height, "-") for _ in NUM_SQ_ADD])
                dim = length(vec_height) + 1
                vec_rank = dim .+ NUM_SQ_ADD
                append!(RANK, vec_rank)
                # create the coordinate ring
                coord_ring = build_ring_scroll(vec_height)
            elseif EXAMPLE[idx] == "cubic"
                deg = CUBIC_CURVE_DEG[IDX2PAR[idx]]
                if IDX2PAR[idx] == 1
                    curve_coeff, coord_ring = generate_plane_cubic(deg) # smoothness is checked here
                    # construct its string name for output
                    str_curve = "Cubic:"
                    for i=0:3,j=0:(3-i)
                        if curve_coeff[[i,j]] != 0
                            if curve_coeff[[i,j]] > 0 && i+j > 0
                                str_curve *= "+"
                            end
                            str_curve *= string(curve_coeff[[i,j]])*"x"*string(i)*"y"*string(j)
                        end
                    end
                else
                    coord_ring = build_ring_from_plane_curve(curve_coeff, deg, check_smooth=false)
                end
                # create the name tag from the cubic coefficients and target degree
                append!(NAME, [str_curve*":"*string(deg) for _ in NUM_SQ_ADD])
                vec_rank = 3 .+ NUM_SQ_ADD
                append!(RANK, vec_rank)
            elseif EXAMPLE[idx] == "Veronese"
                dim, deg = VERONESE_DIM_DEG[IDX2PAR[idx]]
                # create the name tag from the heights
                append!(NAME, ["Veronese:"*join([dim, deg], "-") for _ in NUM_SQ_MULT])
                coord_ring = build_ring_Veronese(dim, deg)
                num_BP_bound = get_BP_bound(coord_ring)
                vec_rank = ceil.(Int, num_BP_bound .* NUM_SQ_MULT)
                append!(RANK, vec_rank)
            end
            # execute the experiment
            succ, fail, time, dist, sdpt, sdpr_min, sdpr_med = exec_multiple(coord_ring, vec_rank, flag_comp=true)
            append!(SUCC, succ)
            append!(FAIL, fail)
            append!(TIME, time)
            append!(DIST, dist)
            append!(SDPT, sdpt)
            append!(SDPR_MIN, sdpr_min)
            append!(SDPR_MED, ceil.(Int,sdpr_med))
            # update the output file
            result = DataFrame(:NAME => NAME, 
                               :RANK => RANK, 
                               :SUCC => SUCC, 
                               :TIME => TIME, 
                               :FAIL => FAIL, 
                               :DIST => DIST,
                               :SDPT => SDPT,
                               :SDPR_MIN => SDPR_MIN,
                               :SDPR_MED => SDPR_MED)
            CSV.write(str_output, result)
            println()
        end
    end
end

run_experiments()

