using Flux

function build_mlp(input_dim::Int, hidden_size::Int)
    return Chain(
        Dense(input_dim, hidden_size, gelu),   
        Dense(hidden_size, 32, gelu),          
        Dense(32, 1)                           
    )
end