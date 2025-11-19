struct EulerStep{FT}
    dt::FT
    step_function::Function

    function EulerStep(dt::T) where {T}
        function step!(X_next, X_prev, vf)
            X_next .= X_prev .+ dt*vf(X_prev)
            nothing
        end
        new{T}(dt, step!)
    end
end

struct FullExponentialStep{FT}
    dt::FT
    Op1::Matrix{FT}
    Op2::Matrix{FT}
    step_function::Function

    function FullExponentialStep(dt::T, PA::Matrix{T},
         PB::Union{Matrix{T}, Vector{T}}) where {T}
        
        U, S = svd(B)
        Σr = diagm(S[1:100])
        diag(exp(-Σr*dt))

        Op1 = U[:,1:100]*exp.(-Σr*dt)*U[:,1:100]'
        Op2 = U[:,1:100]*(inv(Σr)*(float.(I(length(100))) .- exp.(-Σr*dt)))*U[:,1:100]'

        function step!(X_next, X_prev)
            X_next .= Op1*X_prev .+ Op2*PB 
            nothing
        end
        new{T}(dt, Op1, Op2, step!)
    end
end

using LinearAlgebra
using Test
TEST = false
if TEST
    A = rand(10,10)
    B = rand(10, 40)
    P = float.(Array(I(10)))
    gf = GradientFlow(A, B, P)
    dt = 1.0
    scheme = EulerStep(dt)
    iterator = scheme.step_function
    x_new = rand(10, 40)
    x_old = rand(10, 40)
    vf = gf.vf
    @test iterator(x_new, x_old, vf) == nothing
end

if TEST
    A = rand(10,10)
    B = rand(10, 40)
    P = float.(Array(I(10)))
    gf = GradientFlow(A, B, P)
    dt = 1.0
    scheme = FullExponentialStep(dt, P*A, P*B)
    iterator = scheme.step_function    
    x_new = rand(10, 40)
    x_old = rand(10, 40)
    @test iterator(x_new, x_old) == nothing
end