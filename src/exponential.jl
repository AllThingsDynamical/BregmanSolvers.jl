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