# LowRankSOS.jl

Low-rank nonconvex formulation for finding sum-of-square representation of a given quadratic form over a real projective variety.
Current examples include rational normal scrolls (scroll), plane cubic curves (cubic), and Veronese varieties (Veronese).
## Run Experiments
Experiments can be run through the `run_experiment.jl` script under `example/`, with the options
* [example name] which can be `scroll`, `cubic`, `Veronese`, and `all` (default)
* [number of repetition] which is the number that we repeat the experiment runs for each setting, default to $100$
* [output file name] which is name for the output file, e.g., `cubic` will lead to the output file being named `result_cubic.csv`

Before official release, the current recommended practice of running the experiments from command line interface is the following, assuming the working directory contains `LowRankSOS.jl`.
```
julia --project=LowRankSOS.jl/. LowRankSOS.jl/example/run_experiment.jl [example name] [number of repetition] [output file name]
```
