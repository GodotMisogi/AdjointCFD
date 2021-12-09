## Optimization setup
 
# Derivative stuff for learning. Consider OpenMDAO to use MAUD generalizations.
solve_direct(x, p, ∂R∂x, ∂R∂p) = @views reshape(reduce(hcat, ∂R∂p \ -(∂R∂x)[:,:,i][:] for i in eachindex(x)), (size(p)..., length(x)))

solve_adjoint(f, p, ∂R∂p, dfdp) = @views reshape(reduce(hcat, ∂R∂p' \ -(dfdp)'[i,:] for i in eachindex(f)), (length(f), size(p)...)

total_derivative_direct(∂f∂x, ψ, ∂f∂p) = @views ∂f∂x + [ sum(∂f∂p * ψ[n]) for n in eachindex(∂f∂x) ]

total_derivative_adjoint(∂f∂x, φ, ∂R∂x) = @views ∂f∂x + [ sum(permutedims(φ) * reshape(∂R∂x, (length(R[:]), length(∂f∂x)))[:,n]) for n in eachindex(∂f∂x) ]