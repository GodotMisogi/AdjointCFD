using LinearAlgebra
using Zygote
using ForwardDiff
using ReverseDiff
using StaticArrays
using Base.Iterators
using Plots

## Stencil generation
neighbours(CI, deg)        = CI .+ CartesianIndices(ntuple(x -> -deg:deg, length(CI)))

stencil_depth(CI, depth)   = [ CI + CartesianIndex(ntuple(x -> ifelse(x == i, depth, 0), length(CI))) for i in 1:length(CI) ]

stencils(CI, deg, poke = false) = ifelse(poke, [ stencil_depth(CI, -(deg + 1)); neighbours(CI, deg)[:]; stencil_depth(CI, deg + 1) ], neighbours(CI, deg)[:])

von_neumann_stencil(CI, n) = stencils(CI, n, true)
moore_stencil(CI, n)       = stencils(CI, n, false)
cs_stencil(φ, index)       = @view φ[von_neumann_stencil(index, 0)]

## Difference operators specific to CS stencil

# Finite differencing operations on values
central_2nd(φp, φ, φm)     = φm - 2φ + φp
Δ²_central(φp, φ, φm, Δx)  = central_2nd(φp, φ, φm) / Δx^2
Δ_central(φp, φm, Δx)      = (φp - φm) / 2Δx
Δ_forward(φ1, φ2, Δt)      = (φ2 - φ1) / Δt

# First derivatives
Δx_central(φ, ps)  = Δ_central(φ[4], φ[1], norm(ps[4] .- ps[1]))
Δy_central(φ, ps)  = Δ_central(φ[5], φ[2], norm(ps[5] .- ps[2]))

Δx_backward(φ, ps) = Δ_backward(φ[3], φ[1], norm(ps[3] .- ps[1]))
Δy_backward(φ, ps) = Δ_backward(φ[3], φ[2], norm(ps[3] .- ps[2]))

Δx_forward(φ, ps)  = Δ_forward(φ[4], φ[3], norm(ps[4] .- ps[3]))
Δy_forward(φ, ps)  = Δ_forward(φ[5], φ[3], norm(ps[5] .- ps[3]))

# Second derivatives
Δ²x_central(φ, ps) = Δ²_central(φ[4], φ[3], φ[1], norm(ps[4] .- ps[1]))
Δ²y_central(φ, ps) = Δ²_central(φ[5], φ[3], φ[2], norm(ps[5] .- ps[2]))

# Boundary
function substitute(φ, index, val)
    if index ∈ CartesianIndices(φ)
        φ[index]
    else
        val
    end
end

## Governing equations and residual setup
temperature_residual(φ, k, q, pts) = -k * (Δ²x_central(φ, pts) + Δ²y_central(φ, pts)) + q

# Ruleset for thermal conduction with a source field
function ruleset(φs, σs, k, grid, index, φ_boundaries)
    # Resize fields to grid
    φs = reshape(φs, size(grid)...)
    σs = reshape(σs, size(grid)...)

    # Boundary conditions check
    if any(index.I .<= 1) || any(index.I .>= size(φs))
        # There have to be more conditions generally...
        φs_vec = [ substitute(φs, local_index, φ_boundaries) for local_index in von_neumann_stencil(index, 0) ]
        ds_vec = [ substitute(grid, local_index, 0.1)        for local_index in von_neumann_stencil(index, 0) ]
    else
        # Get neighbours of a cell and corresponding sizes
        φs_vec = cs_stencil(φs, index)
        ds_vec = cs_stencil(grid, index)
    end

    σ = σs[index]

    # Evaluate residuals
    temperature_residual(φs_vec, k, σ, ds_vec)
end

# Compute residual equations
compute_residuals(p, σ, k, grid, φ_boundary) = [ ruleset(p, σ, k, grid, index, φ_boundary) for index in CartesianIndices(grid) ]

newton_update(p, δp, α = 1.0) = p - α * δp

# Pre-allocated version
newton_update!(p, δp, α = 1.0) = @. p = p - α * δp

## Initial setup
nx, ny  = 50, 40

# Less allocations than broadcasting inside sum
polynomial(x, coeffs) = sum(k -> k[1] * x^k[2], zip(coeffs, 0:length(coeffs)-1))

# Grid generation
xs      = range(0, 1, length = nx)
ys      = polynomial.(xs, Ref(ones(5)))
grid    = reshape([ (x, y) for (x, y_max) in zip(xs, ys) for y in range(y_max, -y_max, length = 2ny) ], 2ny, nx)

##
α       = 1.0
k       = 1.
φ_bound = 200

φ0      = φ_bound .* ones(size(grid)...)
σ       = rand(size(grid)...)

p       = φ0[:]
R       = similar(p)
# ∂R∂p = zeros(2 .* (prod(size(ωs)), prod((reverse ∘ size)(ωs)))...)

num_iters = 3
compute_residuals!(R, p) = R .= compute_residuals(p, σ, k, grid, φ_bound)[:]
compute_grad_fwd(R, p) = ForwardDiff.jacobian(compute_residuals!, R, p)

##
# using NLsolve
# prob = nlsolve(compute_residuals!, p)

## NEWTON ITERATIONS
#=================================================================#

function newton_solver!(R, p, num_iters = 3, α = 1.0)
    ε = zeros(num_iters);
    for i in 1:num_iters
        R    = compute_residuals!(R, p)
        ∂R∂p = compute_grad_fwd(R, p)
        Δp   = ∂R∂p \ -R
        p   .= newton_update!(p, Δp, α)

        # Error processing
        ε[i] = maximum(abs.(Δp))
        println("Newton step error: $(ε[i])")
    end
    R, p, ε
end

##
R, p, ε = newton_solver!(R, p, num_iters, α)


## Optimization setup

# Derivative stuff
function solve_direct(x, p, ∂R∂x, ∂R∂p)
    ∂R∂p_sq = reshape(∂R∂p, (length(p[:]), length(p[:])))
    reshape(hcat((∂R∂p_sq \ -(∂R∂x)[:,:,i][:] for i in eachindex(x))...), (size(p)..., length(x)))
end

function solve_adjoint(p, ∂R∂p, dfdp) 
    reshape(∂R∂p, (length(p[:]), length(p[:])))' \ -(dfdp)'[:]
end

total_derivative_direct(∂f∂x, ψ, ∂f∂p) = ∂f∂x + [ sum(∂f∂p * ψ[n]) for n in eachindex(∂f∂x) ]

total_derivative_adjoint(∂f∂x, φ, ∂R∂x) = ∂f∂x + [ sum(permutedims(φ) * reshape(∂R∂x, (length(R[:]), length(∂f∂x)))[:,n]) for n in eachindex(∂f∂x) ]