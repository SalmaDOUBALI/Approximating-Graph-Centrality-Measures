using Flux, Wandb, Printf

function train_model(config, train_data, test_data)
    # Initialize Wandb Logger
    lg = WandbLogger(project="Approximating_Centrality", name=config["run_name"], config=config)

    # Prepare Data
    X_train = hcat([d.X for d in train_data]...)
    Y_train = hcat([d.Y for d in train_data]...)
    loader = Flux.DataLoader((X_train, Y_train), batchsize=config["batch_size"], shuffle=true)

    # Setup Model: Designing lightweight MLP
    model = build_mlp(1, config["hidden_size"])
    opt_state = Flux.setup(Flux.Adam(config["lr"]), model)

    println("Starting training...")
    for epoch in 1:config["epochs"]
        total_loss = 0.0
        for (x, y) in loader
            loss, grads = Flux.withgradient(model) do m
                Flux.logitcrossentropy(m(x), y)
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += loss
        end

        # --- EVALUATION SECTION ---
        # Evaluation on the first test graph to observe generalizability
        test_sample = test_data[1]
        
        # Call the updated evaluation function
        results = evaluate_performance(model, test_sample.X, test_sample.Y, test_sample.bc_scores, 10)

        # Log ALL metrics to Wandb including Kendall's Tau
        Wandb.log(lg, Dict(
            "loss" => total_loss / length(loader),
            "accuracy" => results.accuracy,
            "precision_at_10" => results.precision_k,
            "spearman_correlation" => results.spearman,
            "kendall_tau" => results.kendall,        # Added to Wandb!
            "inference_time_ms" => results.time_ms,
            "epoch" => epoch
        ))
        
        if epoch % 10 == 0
             @printf("Epoch %d | Loss: %.4f | Prec@10: %.2f | Spearman: %.2f | Kendall: %.2f\n", 
                     epoch, total_loss/length(loader), results.precision_k, results.spearman, results.kendall)
        end
    end
    
    Wandb.close(lg)
    return model
end