# Optional dependency: routes Tortuosity's HDF5 hook to HDF5.jl.
module TortuosityHDF5Ext

using HDF5
using Tortuosity

Tortuosity._h5open(f::Function, fname, mode) = HDF5.h5open(f, fname, mode)

end
