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
const NUM_REPEAT_LARGE = 5
const REL_MAX_ITER = 100
const SCROLL_HEIGHTS = [[5,10],[10,15],[15,20],
                        [30,40],[50,60],[70,80],
                        [5,10,15],[10,20,30],[20,30,40],
                        [5,10,15,20],[10,15,20,25],
                        [15,20,25,30],[5,10,15,20,25]]
const SCROLL_HEIGHTS_LARGE = [[50,100],[100,200],[200,300],
                              [300,400],[500,600],[700,800]]
const CUBIC_CURVE_DEG = [10,20,30,40,50]
const CUBIC_CURVE_DEG_LARGE = [100,200,300,400,500]
const VERONESE_DIM_DEG = [(4,2),(6,2),(8,2),(10,2),
                          (3,3),(4,3),(5,3),
                          (2,4),(3,4),(2,5)]

const NUM_SQ_ADD = [0,1,2,3]
const NUM_SQ_MULT = [1.0,1.1,1.2,1.5,2.0]
const SOLVER_COMP = ["CSDP", "SCS", "Hypatia"] # Clarabel causes OOM on Slurm Clusters

# include functions that execute single or multiple experiments
include("exec_single.jl")
include("exec_multiple.jl")


# check the command line arguments
num_args = length(ARGS)
## by default we run experiments on all examples
# allowed options: "small", "scroll", "cubic", or "Veronese"
# and "large", "scroll-large", "cubic-large"
str_example = "small"
if num_args > 0
    str_example = ARGS[1]
end
## by default we run batch experiments and retrieve the statistics
### when `num_repeat == 1` we only run the experiment with the first parameter in the arrays
num_repeat = NUM_REPEAT
if str_example in ["large", "scroll-large", "cubic_large"]
    num_repeat = NUM_REPEAT_LARGE
end
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
        if str_example == "scroll" || str_example == "small"
            vec_height = SCROLL_HEIGHTS[1]
            dim = length(vec_height)
            coord_ring = build_ring_scroll(vec_height)
            exec_single(coord_ring, dim+1)
        end
        if str_example == "cubic" || str_example == "small"
            deg_target = CUBIC_CURVE_DEG[1]
            _, coord_ring = generate_plane_cubic(deg_target)
            exec_single(coord_ring, 3)
        end
        if str_example == "Veronese" || str_example == "small"
            dim, deg = VERONESE_DIM_DEG[1]
            coord_ring = build_ring_Veronese(dim, deg)
            num_sq = get_BP_bound(coord_ring)
            exec_single(coord_ring, num_sq)
        end
    else
        # prepare the inputs
        EXAMPLE = String[]
        IDX2PAR = Int[]
        if str_example == "scroll" || str_example == "small"
            append!(EXAMPLE, ["scroll" for _ in 1:length(SCROLL_HEIGHTS)])
            append!(IDX2PAR, collect(1:length(SCROLL_HEIGHTS)))
        end
        if str_example == "cubic" || str_example == "small"
            append!(EXAMPLE, ["cubic" for _ in 1:length(CUBIC_CURVE_DEG)])
            append!(IDX2PAR, collect(1:length(CUBIC_CURVE_DEG)))
        end
        if str_example == "Veronese" || str_example == "small"
            append!(EXAMPLE, ["Veronese" for _ in 1:length(VERONESE_DIM_DEG)])
            append!(IDX2PAR, collect(1:length(VERONESE_DIM_DEG)))
        end
        if str_example == "scroll-large" || str_example == "large"
            append!(EXAMPLE, ["scroll-large" for _ in 1:length(SCROLL_HEIGHTS_LARGE)])
            append!(IDX2PAR, collect(1:length(SCROLL_HEIGHTS_LARGE)))
        end
        if str_example == "cubic-large" || str_example == "large"
            append!(EXAMPLE, ["cubic-large" for _ in 1:length(CUBIC_CURVE_DEG_LARGE)])
            append!(IDX2PAR, collect(1:length(CUBIC_CURVE_DEG_LARGE)))
        end
        # prepare the outputs
        NAME = String[]
        RANK = Int[]
        SUCC = Int[]
        FAIL = Int[]
        TIME = Float64[]
        DIST = Float64[]
        SDPTIME = Dict{String,Vector{Float64}}()
        SDPRANK_MIN = Dict{String, Vector{Int}}()
        SDPRANK_MED = Dict{String, Vector{Int}}()
        for name in SOLVER_COMP
            SDPTIME[name] = Float64[]
            SDPRANK_MIN[name] = Int[]
            SDPRANK_MED[name] = Int[]
        end
        # start the main experiments
        num_experiment = length(EXAMPLE)
        vec_rank = Int[]
        coord_ring = CoordinateRing2(0,0,SparseVector{Rational{Int},Int}[])
        curve_coeff = Dict{Vector{Int},Int}() # for cubic curves only
        str_curve = ""
        for idx = 1:num_experiment
            if EXAMPLE[idx] == "scroll" || EXAMPLE[idx] == "scroll-large"
                vec_height = []
                if EXAMPLE[idx] == "scroll"
                    vec_height = SCROLL_HEIGHTS[IDX2PAR[idx]]
                    dim = length(vec_height) + 1
                    vec_rank = dim .+ NUM_SQ_ADD
                else
                    vec_height = SCROLL_HEIGHTS_LARGE[IDX2PAR[idx]]
                    dim = length(vec_height) + 1
                    vec_rank = [dim]
                end
                # create the name tag from the heights
                append!(NAME, ["Scroll:"*join(vec_height, "-") for _ in vec_rank])
                append!(RANK, vec_rank)
                # create the coordinate ring
                coord_ring = build_ring_scroll(vec_height)
            elseif EXAMPLE[idx] == "cubic" || EXAMPLE[idx] == "cubic-large"
                deg = 0
                if EXAMPLE[idx] == "cubic"
                    deg = CUBIC_CURVE_DEG[IDX2PAR[idx]]
                    vec_rank = 3 .+ NUM_SQ_ADD
                else
                    deg = CUBIC_CURVE_DEG_LARGE[IDX2PAR[idx]]
                    vec_rank = [3]
                end
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
                append!(NAME, [str_curve*":"*string(deg) for _ in vec_rank])
                append!(RANK, vec_rank)
            elseif EXAMPLE[idx] == "Veronese"
                dim, deg = VERONESE_DIM_DEG[IDX2PAR[idx]]
                # create the name tag from the heights
                vec_rank = ceil.(Int, num_BP_bound .* NUM_SQ_MULT)
                append!(NAME, ["Veronese:"*join([dim, deg], "-") for _ in vec_rank])
                append!(RANK, vec_rank)
                coord_ring = build_ring_Veronese(dim, deg)
                num_BP_bound = get_BP_bound(coord_ring)
            end
            # execute the experiment
            succ, fail, time, dist, sdptime, sdprank_min, sdprank_med = exec_multiple(coord_ring, vec_rank, solver_comp=SOLVER_COMP)
            append!(SUCC, succ)
            append!(FAIL, fail)
            append!(TIME, time)
            append!(DIST, dist)
            for name in SOLVER_COMP
                append!(SDPTIME[name], sdptime[name])
                append!(SDPRANK_MIN[name], sdprank_min[name])
                append!(SDPRANK_MED[name], ceil.(Int,sdprank_med[name]))
            end
            # update the output file
            result = ["NAME" => NAME,
                      "RANK" => RANK,
                      "SUCC" => SUCC,
                      "TIME" => TIME,
                      "FAIL" => FAIL,
                      "DIST" => DIST]
            for name in SOLVER_COMP
                append!(result, ["TIME-"*name => SDPTIME[name],
                                 "RANK-MIN-"*name => SDPRANK_MIN[name],
                                 "RANK-MED-"*name => SDPRANK_MED[name]])
            end
            CSV.write(str_output, DataFrame(result))
            println()
        end
    end
end

run_experiments()

