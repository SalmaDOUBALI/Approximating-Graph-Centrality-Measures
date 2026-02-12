using Graphs, JLD2, FileIO, Statistics, StatsBase, Flux

include("../src/Model_MLP.jl")
include("../src/evaluate.jl")

println("Loading the classification MLP Model...")
model_data = load("data/model_combined_top1.jld2")
model = model_data["model"]

function load_email_eu(filepath)
    g = SimpleGraph()
    max_id = 0
    
    open(filepath) do file
        for line in eachline(file)
            if !startswith(line, "#")
                u, v = parse.(Int, split(line))
                max_id = max(max_id, u, v)
            end
        end
    end
    
    add_vertices!(g, max_id + 1) 
    
    open(filepath) do file
        for line in eachline(file)
            if !startswith(line, "#")
                u, v = parse.(Int, split(line))
                add_edge!(g, u + 1, v + 1)
            end
        end
    end
    return g
end

println("Loading Email-Eu-core dataset...")
g_real = load_email_eu("data/email-Eu-core.txt")
n = nv(g_real)
println("Graph loaded: $n Nodes, $(ne(g_real)) Edges.")

println("Computing BC with Brand's Algo...")
bc_exact = betweenness_centrality(g_real)

deg = degree(g_real)
max_d = max(deg...)
X_real = Float32.(reshape(deg ./ max_d, 1, :)) 

k = max(1, Int(round(0.01 * n)))

results = evaluate_performance(model, X_real, nothing, bc_exact, k)

println("\nResults for classification MLP Model")
println("Spearman: ", round(results.spearman; digits=4))
println("Top-1%:   ", round(results.precision_k * 100; digits=2), "%")
println("Time:    ", round(results.time_ms; digits=2), " ms")