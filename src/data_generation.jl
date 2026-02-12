using Graphs, Random, Statistics, JLD2
using Base.Threads

function generate_graph_dataset(type::String, n_graphs::Int, n_nodes::Int, top_k::Int; prob=0.02)

    dataset = Vector{Any}(undef, n_graphs) 
    println("Generating $n_graphs graphs of type $type using $(nthreads()) threads...")

    @threads for i in 1:n_graphs
        g = if type == "ER"
            erdos_renyi(n_nodes, prob)
        elseif type == "BA"
            barabasi_albert(n_nodes, 3) 
        elseif type == "NWS"
            newman_watts_strogatz(n_nodes, 4, 0.1) 
        elseif type == "SSF"
            static_scale_free(n_nodes, n_nodes*2, 2.5) 
        else
            error("Unknown graph type: $type")
        end    
            
        #normalizing the degree 
        deg = degree(g)
        max_d = maximum(deg)
        X = reshape(Float32.(max_d > 0 ? deg ./ max_d : deg), 1, :)

        bc_scores = betweenness_centrality(g)
        
        scores_sorted = sort(bc_scores, rev=true)

        k_1_percent = max(1, Int(floor(n_nodes * 0.01)))
        threshold = scores_sorted[k_1_percent]

        is_top_k = bc_scores .>= threshold
        Y = Flux.onehotbatch(is_top_k, [true, false])
        
        dataset[i] = (X=X, Y=Y, bc_scores=bc_scores, g=g)        
        if i % 50 == 0 
            println("  -> Thread $(threadid()) processed graph $i") 
        end
    end
    return dataset
end