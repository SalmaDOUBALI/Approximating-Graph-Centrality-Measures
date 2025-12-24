using Flux

"""
    build_mlp(input_dim::Int, hidden_dim::Int)
Creates a simple MLP for centrality approximation. [cite: 44, 53]
"""
function build_mlp(input_dim::Int, hidden_dim::Int)
    return Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => 2) # Output: [is_top_k, is_not_top_k]
    )
end