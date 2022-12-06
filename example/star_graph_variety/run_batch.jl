include("./star_graph.jl")

dim = 20
mat_aux = randn(dim,dim)
mat_target = mat_aux' * mat_aux

num_sample = 1000
num_iter = 3000

test_batch_on_star_graph(dim, mat_target=mat_target,str_method="gradient", num_sample=num_sample, num_max_iter=num_iter)
test_batch_on_star_graph(dim, mat_target=mat_target,str_method="pushforward", num_sample=num_sample, num_max_iter=num_iter)
