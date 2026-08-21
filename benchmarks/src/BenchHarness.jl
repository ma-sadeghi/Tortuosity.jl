# Shared machinery for the Tortuosity.jl benchmark campaign: the case grid, the
# image store, resumable result files and memory instrumentation.
#
# Included by the CLI entry points in the parent directory rather than installed
# as a package, so `julia --project=. <script>.jl` needs no extra setup. The
# Python half of the harness (`benchkit/`) mirrors this file for file formats and
# reads the same `config.toml`, so the two never disagree about what a case is.

module BenchHarness

using Dates
using HDF5
using Printf
using SHA
using Statistics
using TOML

export Case, Config
export load_config, select_cases, case_grid
export image_path, ensure_image!, load_image, read_manifest, manifest_path
export references_path, read_references, record_reference
export ResultsWriter, write_row!, completed_cases, close_writer!
export current_rss, device_live_bytes, device_pool_bytes, with_peak_sampling
export parse_args, flag, option, record_environment, benchdir

include("config.jl")
include("images.jl")
include("results.jl")
include("memory.jl")
include("cli.jl")

end  # module BenchHarness
