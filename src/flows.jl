struct GradientFlow{FT, ST}
    A::Matrix{FT}
    B::ST
    P::Matrix{FT}
    vf::Any
    
    function GradientFlow(A::Matrix, B::T, P::Matrix) where {T}
        vf = function(X::Union{Matrix, Vector})
            P*B - P*A*X
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