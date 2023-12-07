# the following data lead to a stationary point on the 2-block rational normal scroll

vec_deg = [3, 4]
dim = sum(vec_deg) + length(vec_deg)
rank = length(vec_deg) + 1

mat_start = zeros(rank, dim)
for i = 1:rank
    mat_start[i,i] = 1.0
end
mat_target = mat_start' * mat_start
mat_target[end,end] += 1.0

# add some perturbation to the start (optional)
mat_start[1, vec_deg[1]+2] += 0.1
