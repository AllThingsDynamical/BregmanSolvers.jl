module BregmanFlows
    using LinearAlgebra
    using SparseArrays
    
    include("flows.jl")

    include("exponential.jl")
    include("exponential_krylov.jl")
    include("multigrid.jl")
    include("domain_decomposition.jl")
    include("dlra.jl")
end