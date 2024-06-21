include("./star_graph.jl")

# run the star graph example with random starts and targets
include("./data/local_min.jl")
test_star_graph(dim, mat_start=mat_start, mat_target=mat_target)
