# the following problem data show that the stationary point can happen 
# in the interior of the SOS cone

dim = 6
# choose a target quadratic form corresponding to x⁴+x²y²+y⁴+z⁴
mat_target = zeros(dim,dim)
mat_target[1,1] = 1
mat_target[2,2] = 1
mat_target[4,4] = 1
mat_target[6,6] = 1
# set the number of squares and starting point corresponding to the tuple (x²,xy,y²)
mat_start = [1.0 0.0 0.0 0.0 0.0 0.0;
             0.0 0.9 0.0 0.0 0.0 0.0;
             0.0 0.0 0.0 1.0 0.1 0.0]
