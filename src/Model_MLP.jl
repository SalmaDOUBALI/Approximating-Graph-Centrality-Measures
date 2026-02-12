using Flux

function build_mlp(input_dim::Int, hidden_dim::Int)
    return Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => 2) 
    )
end