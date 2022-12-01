# problem data for a local minimum on the star graph variety
dim = 20
mat_start = [1.0 zeros(1,dim-1); 0.0 1.0 zeros(1,dim-2)] # the number of squares is 2
mat_target = mat_start' * mat_start
mat_target[dim,dim] += 1.0
# add below some perturbation to see the attraction of this local minimum (optional)
for i = 3:(dim-1)
    mat_start[1,i] += 0.1
end
