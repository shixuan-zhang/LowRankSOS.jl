include("./rational_normal_scroll.jl")

# vec_deg = [1, 1, 1]
vec_deg = [2, 3, 4, 5, 6]
dim = sum(vec_deg) + length(vec_deg)
mat_aux = randn(dim,dim)
mat_target = mat_aux' * mat_aux

# include("./data/stat_point_3block_height1.jl")

num_sample = 1000
num_iter = 3000

test_batch_on_scroll(vec_deg, mat_target=mat_target,str_method="gradient", num_sample=num_sample, num_max_iter=num_iter)
test_batch_on_scroll(vec_deg, mat_target=mat_target,str_method="pushforward", num_sample=num_sample, num_max_iter=num_iter)
test_batch_on_scroll(vec_deg, mat_target=mat_target,str_method="gradient+bypass", num_sample=num_sample, num_max_iter=num_iter)
test_batch_on_scroll(vec_deg, mat_target=mat_target,str_method="pushforward+bypass", num_sample=num_sample, num_max_iter=num_iter)
