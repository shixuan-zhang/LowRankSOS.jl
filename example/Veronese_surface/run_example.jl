include("./Veronese_surface.jl")

include("./data/stat_point.jl")
# include("./data/push_stat_point.jl")
# include("./data/grad_int_stat_point.jl")
compare_methods_on_Veronese_surface(mat_start=mat_start, mat_target=mat_target)
