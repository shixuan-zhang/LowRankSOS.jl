# the following problem data show that the pushforward descent method 
# could also get stuck at a stationary point

dim = 6
# choose a target quadratic form corresponding to x⁴+x²y²+y⁴+z⁴
mat_target = zeros(dim,dim)
mat_target[1,1] = 1
mat_target[2,2] = 1
mat_target[4,4] = 1
mat_target[6,6] = 1
# choose a starting point 
mat_start = [1.0 0.0 0.0 0.0 0.0 0.0;
             0.0 2.0 0.0 0.0 0.0 0.0;
             0.0 -.1 -.1 3.0 0.0 0.0]
