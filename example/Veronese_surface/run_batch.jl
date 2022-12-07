include("./Veronese_surface.jl")

dim = 6
mat_aux = randn(dim,dim)
mat_target = mat_aux' * mat_aux

num_sample = 1000
num_iter = 3000

test_batch_on_Veronese_surface(mat_target=mat_target,str_method="gradient", num_sample=num_sample, num_max_iter=num_iter)
test_batch_on_Veronese_surface(mat_target=mat_target,str_method="pushforward", num_sample=num_sample, num_max_iter=num_iter)
