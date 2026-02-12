using Flux, Wandb, Printf, Statistics, Random

function ranking_loss(model, x_i, x_j)
    scores_i = model(x_i)
    scores_j = model(x_j)
    
    diff = scores_j .- scores_i
    return mean(log.(1 .+ exp.(diff)))
end

function train_model(config, train_data, test_data)
    lg = WandbLogger(project="Approximating_Centrality", name=config["run_name"], config=config)

    model = build_mlp(1, config["hidden_size"]) 
    opt_state = Flux.setup(Flux.Adam(config["lr"]), model)

    println("Starting training...")
    
    for epoch in 1:config["epochs"]
        total_loss = 0.0
        
        shuffle!(train_data)

        for sample in train_data
            n_nodes = length(sample.bc_scores)
            idx_i = rand(1:n_nodes, 500) 
            idx_j = rand(1:n_nodes, 500)
            
            mask = sample.bc_scores[idx_i] .> sample.bc_scores[idx_j]
            
            x_i = sample.X[:, idx_i[mask]]
            x_j = sample.X[:, idx_j[mask]]

            if isempty(x_i)
                continue
            end

            loss, grads = Flux.withgradient(model) do m
                ranking_loss(m, x_i, x_j)
            end
            
            Flux.update!(opt_state, model, grads[1])
            total_loss += loss
        end

        test_sample = test_data[1]
        results = evaluate_performance(model, test_sample.X, test_sample.Y, test_sample.bc_scores, 10)

        # Log metrics on Wandb
        Wandb.log(lg, Dict(
            "loss" => total_loss / length(train_data),
            "top_1_percent_acc" => results.precision_k,
            "spearman_correlation" => results.spearman,
            "kendall_tau" => results.kendall,        
            "inference_time_ms" => results.time_ms,
            "epoch" => epoch
        ))
        
        if epoch % 5 == 0
             @printf("Epoch %d | Loss: %.4f | Spearman: %.2f | Top1%%: %.2f\n", 
                     epoch, total_loss/length(train_data), results.spearman, results.precision_k)
        end
    end
    
    Wandb.close(lg)
    return model
end