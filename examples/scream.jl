using LinearAlgebra
using Zygote
using ForwardDiff
using ReverseDiff
using DiffEqOperators
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

## Cells and boundaries
abstract type AbstractCell end

struct CartesianPoint{M,N,T} <: AbstractCell
    index  :: CartesianIndex{N}
    values :: SVector{M,T}
end

von_neumann_stencil(ci :: CartesianPoint, n) = von_neumann_stencil(ci.index, n)
moore_stencil(ci :: CartesianPoint, n)       = moore_stencil(ci.index, n)

struct CartesianBoundary{M,N,T} <: AbstractCell
    point      :: CartesianPoint{M,N,T}
    neighbours :: Vector{Tuple{CartesianIndex{N},SVector{M,T}}}
end

## Finite differencing operations on values and stencils

# First derivatives
Δ_central(φp, φm, Δx)     = (φp - φm) / 2Δx
Δ_forward(φ1, φ2, Δt)     = (φ2 - φ1) / Δt
central_2nd(φp, φ, φm)    = φm - 2φ + φp

# Stencil
Δx_central(φ, dx)  = Δ_central(φ[4], φ[1], dx)
Δy_central(φ, dy)  = Δ_central(φ[5], φ[2], dy)

Δx_backward(φ, dx) = Δ_backward(φ[3], φ[1], dx)
Δy_backward(φ, dy) = Δ_backward(φ[3], φ[2], dy)

Δx_forward(φ, dx)  = Δ_forward(φ[4], φ[3], dx)
Δy_forward(φ, dy)  = Δ_forward(φ[5], φ[3], dy)

# Second derivatives
Δ²_central(φp, φ, φm, Δx) = central_2nd(φp, φ, φm) / Δx^2
Δ²_central(φp, φ, φm, Δxp, Δxm) = Δ_forward(φp, φ, Δxp) - Δ_forward(φ, φm, Δxm)

# Stencil
Δ²x_central(φ, dx) = Δ²_central(φ[4], φ[3], φ[1], dx)
Δ²y_central(φ, dy) = Δ²_central(φ[5], φ[3], φ[2], dy)

Δ²x_central(φ, dxs) = Δ²_central(φ[4], φ[3], φ[1], dxs[1], dxs[2])
Δ²y_central(φ, dys) = Δ²_central(φ[5], φ[3], φ[2], dys[1], dys[2])

# Boundary (this needs type distinctions between boundary and non-boundary cells to be efficient)
function substitute(φ, index, val)
    if index ∈ CartesianIndices(φ)
        φ[index]
    else
        val
    end
end

## Governing equations and residual setup
temperature_residual(φ, k, q, ds) = -k * (Δ²x_central(φ, ds[1:2]) + Δ²y_central(φ, ds[3:4])) + q

# Ruleset for thermal diffusion with a source field
function ruleset(φs, σs, k, dxs, dys, index, φ_boundaries)
    # Boundary conditions check
    if any(index.I .<= 1) || any(index.I .>= size(φs))
        # There have to be more conditions generally...
        φs_vec = [ substitute(φs, local_index, φ_boundaries) for local_index in von_neumann_stencil(index, 0) ]
        ds_vec = [ substitute(dxs, local_index, 0.1)          for local_index in von_neumann_stencil(index, 0) ]
    else
        # Get neighbours of a cell and corresponding sizes
        φs_vec   = cs_stencil(φs, index)
        dxp, dxn = dxs[index], dxs[index + CartesianIndex(1,0)]
        dyp, dyn = dys[index], dys[index + CartesianIndex(0,1)]
        ds_vec   = [ dxp; dxn; dyp; dyn ]
    end

    σ = σs[index]

    # Evaluate residuals
    temperature_residual(φs_vec, k, σ, ds_vec)
end

# Compute residual equations
compute_residuals(p, σ, k, dxs, dys, φ_boundary) = map(index -> ruleset(p, σ, k, dxs, dys, index, φ_boundary), CartesianIndices(p))

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
grid    = reshape([ SVector(x, y) for (x, y_max) in zip(xs, ys) for y in range(y_max, -y_max, length = 2ny) ], 2ny, nx)

# Compute grid spacings
dxs = norm.(grid[:,2:end] .- grid[:,1:end-1])
dys = norm.(grid[2:end,:] .- grid[1:end-1,:])

##
α       = 1.0
k       = 1.
φ_bound = 200

φ0      = φ_bound .* ones(size(grid)...)
σ       = rand(size(grid)...)

p       = φ0
R       = similar(p[:])
# ∂R∂p = zeros(2 .* (prod(size(ωs)), prod((reverse ∘ size)(ωs)))...)

num_iters = 3
compute_residuals!(R, p) = R .= compute_residuals(p, σ, k, dxs, dys, φ_bound)[:]
compute_grad_fwd(R, p) = ForwardDiff.jacobian(compute_residuals!, R, p)

##
# using NLsolve
# prob = nlsolve(compute_residuals!, p)

## NEWTON ITERATIONS
#=================================================================#

function newton_solver!(R, p, num_iters = 3, α = 1.0)
    # Array to store errors
    ε = zeros(num_iters);

    # Newton iteration loop
    for i in 1:num_iters
        # Compute residuals
        R    = compute_residuals!(R, p)

        # Compute Jacobian
        ∂R∂p = compute_grad_fwd(R, p)

        # Compute Newton step
        Δp   = ∂R∂p \ -R

        # Update state with relaxation factor
        p    = newton_update!(p, reshape(Δp, size(p)), α)

        # Error processing
        ε[i] = maximum(abs.(Δp))
        println("Newton step error L² norm: $(ε[i])")
    end
    R, p, ε
end

##
R, p, ε = newton_solver!(R, p, num_iters, α)


## Optimization setup
 
# Derivative stuff
solve_direct(x, p, ∂R∂x, ∂R∂p) = reshape(reduce(hcat, ∂R∂p \ -(∂R∂x)[:,:,i][:] for i in eachindex(x)), (size(p)..., length(x)))

solve_adjoint(f, p, ∂R∂p, dfdp) = reshape(reduce(hcat, ∂R∂p' \ -(dfdp)'[i,:] for i in eachindex(f)), (length(f), size(p)...)

total_derivative_direct(∂f∂x, ψ, ∂f∂p) = ∂f∂x + [ sum(∂f∂p * ψ[n]) for n in eachindex(∂f∂x) ]

total_derivative_adjoint(∂f∂x, φ, ∂R∂x) = ∂f∂x + [ sum(permutedims(φ) * reshape(∂R∂x, (length(R[:]), length(∂f∂x)))[:,n]) for n in eachindex(∂f∂x) ]