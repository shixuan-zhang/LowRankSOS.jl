include("./sun_graph.jl")

cycle = 20
clique = 5
dim = cycle * (clique-1)
mat_aux = randn(dim,dim)
mat_target = mat_aux' * mat_aux

num_sample = 1000
num_iter = 1000

test_batch_on_sun_graph(cycle,clique,mat_target=mat_target,str_method="gradient", num_sample=num_sample, num_max_iter=num_iter)
test_batch_on_sun_graph(cycle,clique,mat_target=mat_target,str_method="quasiNewton", num_sample=num_sample, num_max_iter=num_iter)
