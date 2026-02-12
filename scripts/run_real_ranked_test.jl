using Graphs, JLD2, FileIO, Statistics, StatsBase, Flux

include("../src/Model_ranking_MLP.jl")
include("../src/evaluate.jl")

println("Loading ranking MLP Model...")
model_path = "data/model_ranking_combined_top1.jld2" 

if isfile(model_path)
    model = load(model_path, "model")
    println("Model loaded sucessfully")
else
    error("file not found : $model_path")
end

# load the real email dataset
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

println("Loading Email-Eu-core...")
g_real = load_email_eu("data/email-Eu-core.txt")
n = nv(g_real)

println("Computing BC with Brands Algo...")
bc_exact = betweenness_centrality(g_real)

deg = degree(g_real)
X_real = Float32.(reshape(deg ./ max(deg...), 1, :))

println("Evaluating...")
k = max(1, Int(round(0.01 * n))) 

results = evaluate_performance(model, X_real, nothing, bc_exact, k)

println("\n Results for Ranking model")
println("Spearman: ", round(results.spearman; digits=4))
println("Kendall:  ", round(results.kendall; digits=4))
println("Top-1%:   ", round(results.precision_k * 100; digits=2), "%")
println("Time:    ", round(results.time_ms; digits=2), " ms")