# the following data lead to a stationary point on the simplest 3-block rational normal scroll

vec_deg = [1, 1, 1]
dim = sum(vec_deg) + length(vec_deg)
rank = length(vec_deg) + 1

mat_start = [1.0 0.0 0.0 0.0 0.1 0.0;
             0.0 1.0 0.0 0.0 0.0 0.2;
             0.0 0.0 1.0 0.0 0.0 0.0;
             0.0 0.0 0.0 1.0 0.0 0.0]
mat_target = mat_start' * mat_start
mat_target[end,end] += 1.0
