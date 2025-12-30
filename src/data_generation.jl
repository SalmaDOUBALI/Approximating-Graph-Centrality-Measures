using Graphs, Random, Statistics, JLD2

"""
    generate_graph_dataset(type::String, n_graphs::Int, n_nodes::Int, top_k::Int)
Generates synthetic graphs (ER or BA), calculates BC, and prepares features/labels. 
"""
function generate_graph_dataset(type::String, n_graphs::Int, n_nodes::Int, top_k::Int; prob=0.02)
    dataset = []
    println("Generating $n_graphs graphs of type $type...")

    for i in 1:n_graphs
        # Generate graph based on type
        g = if type == "ER"
            erdos_renyi(n_nodes, prob)
        elseif type == "BA"
            barabasi_albert(n_nodes, 3) # m=3 is a standard starting point
        elseif type == "NWS"
            newman_watts_strogatz(n_nodes, 4, 0.1) # k=4 neighbors, p=0.1 rewiring
        elseif type == "SSF"
            static_scale_free(n_nodes, n_nodes*2, 2.5) # 2n edges, gamma=2.5
        else
            error("Unknown graph type: $type")
        end    
            
        # Features: Normalized Degree [cite: 45]
        deg = degree(g)
        max_d = maximum(deg)
        X = reshape(Float32.(max_d > 0 ? deg ./ max_d : deg), 1, :)

        # Target: Exact Betweenness Centrality [cite: 31, 60]
        bc_scores = betweenness_centrality(g)
        
        # Labels: Binary Top-K [cite: 43]
        scores_sorted = sort(bc_scores, rev=true)
        threshold = scores_sorted[min(top_k, n_nodes)]
        is_top_k = bc_scores .>= threshold
        Y = Flux.onehotbatch(is_top_k, [true, false])

        push!(dataset, (X=X, Y=Y, bc_scores=bc_scores, g=g))
        if i % 10 == 0 println("  -> Processed $i/$n_graphs graphs") end
    end
    return dataset
end