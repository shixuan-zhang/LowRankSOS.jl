# LowRankSOS.jl

Low-rank nonconvex formulation for finding sum-of-square representation of a given quadratic form over a real projective variety.
Current examples include rational normal scrolls (scroll), plane cubic curves (cubic), and Veronese varieties (Veronese).
See more details in the [arXiv preprint](https://arxiv.org/abs/2411.02208).
## Run Experiments
Experiments can be run through the `run_experiment.jl` script under `example/`, with the following options:
* [example name] which can be `scroll`, `cubic`, `Veronese`, `small` (default, running all of the previous examples), `scroll-large`, `cubic-large`, and `large` (running both the `scroll-large` and `cubic-large` instances); on the smaller instances comparison will be done against `Hypatia.jl`, `Clarabel.jl`, `SCS`, and `CSDP` SDP solvers, while on the larger instances only `CSDP` will be included in the output file as comparison.
* [number of repetition] which is the number that we repeat the experiment runs for each setting, default to $100$. Please set it to at least $2$ to produce the csv file output. When this is set to $1$, the script will execute different low-rank methods for demonstration **without producing any csv file output**!
* [output file name] which is name for the output file and default to `result_[example name].csv` if omitted. For example, the output file name will be `result_large.csv` if this argument is omitted and the [example name] is set to be `large`.


Before official release, the current recommended practice of running the experiments from command line interface is the following, assuming the working directory contains `LowRankSOS.jl`.
```
julia --project=LowRankSOS.jl/. LowRankSOS.jl/example/run_experiment.jl [example name] [number of repetition] [output file name]
```
For comparison against any generic SDP solver, it is recommended to explicitly specify the number of threads for a fair comparison, which can be done by the following command.
```
export OMP_NUM_THREADS=[number of threads] && export OMP_THREAD_LIMIT=[number of threads] && julia --project=LowRankSOS.jl/. LowRankSOS.jl/example/run_experiment.jl [example name] [number of repetition]
```
