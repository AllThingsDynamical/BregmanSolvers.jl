struct GradientFlow{FT, ST}
    A::Matrix{FT}
    B::ST
    P::Matrix{FT}
    vf::Any
    
    function GradientFlow(A::Matrix, B::T, P::Matrix) where {T}
        vf = function(X::Union{Matrix, Vector}, t::Any=1.0)
            P*B - P*A*X
        end

        new{eltype(A), typeof(B)}(A, B, P, vf)
    end
end

struct NesterovFlow{FT, ST}
    A::Matrix{FT}
    B::ST
    P::Matrix{FT}
    vf::Any

    function NesterovFlow(A::Matrix, B::T, P::Matrix) where {T}
        vf = function(Y::Union{Matrix, Vector}, t::Any)
            n = Int(size(Y,1)/2)
            vcat(Y[n+1:end,:], -(3/t)*Y[1:n,:] + P*B - P*A*Y[1:n,:])
        end

        new{eltype(A), typeof(B)}(A, B, P, vf)
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
    @test size(gf.vf(rand(10,40))) == (10,  40)
    @test gf.vf(zeros(10, 40)) == P*B
end