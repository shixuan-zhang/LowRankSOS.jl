# LowRankSOS.jl

Low-rank nonconvex formulation for finding sum-of-square representation of a given quadratic form over a real projective variety.
Current examples include rational normal scrolls (scroll), plane cubic curves (cubic), and Veronese varieties (Veronese).
## Run Experiments
Experiments can be run through the `run_experiment.jl` script under `example/`, with the options
* [example name] which can be `scroll`, `cubic`, `Veronese`, `small` (default, running all of the previous examples), `scroll-large`, `cubic-large`, and `large` (running both the `scroll-large` and `cubic-large` instances)
* [number of repetition] which is the number that we repeat the experiment runs for each setting, default to $100$
* [output file name] which is name for the output file and default to `result_[example].csv` if omitted. For example, the output file name will be `result_small.csv` if the example name is set to be `small`.

Before official release, the current recommended practice of running the experiments from command line interface is the following, assuming the working directory contains `LowRankSOS.jl`.
```
julia --project=LowRankSOS.jl/. LowRankSOS.jl/example/run_experiment.jl [example name] [number of repetition] [output file name]
```
For comparison against the standard SDP solver, it is recommended to explicitly specify the number of threads for a fair comparison, which can be done by the following command.
```
export OMP_NUM_THREADS=[number of threads] && export OMP_THREAD_LIMIT=[number of threads] && julia --project=LowRankSOS.jl/. LowRankSOS.jl/example/run_experiment.jl [example name] [number of repetition]
```
