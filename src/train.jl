using Flux, Wandb, Printf

function train_model(config, train_data, test_data)
    # Initialize Wandb Logger [cite: 53]
    lg = WandbLogger(project="Approximating_Centrality", name=config["run_name"], config=config)

    # Prepare Data
    X_train = hcat([d.X for d in train_data]...)
    Y_train = hcat([d.Y for d in train_data]...)
    loader = Flux.DataLoader((X_train, Y_train), batchsize=config["batch_size"], shuffle=true)

    # Setup Model: Designing lightweight MLP [cite: 53]
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

        # --- UPDATED SECTION ---
        # Evaluation on the first test graph to observe generalizability [cite: 79]
        test_sample = test_data[1]
        
        # Call the new evaluation function that returns all 4 metrics
        results = evaluate_performance(model, test_sample.X, test_sample.Y, test_sample.bc_scores, 10)

        # Log ALL metrics to Wandb to compare efficiency and accuracy [cite: 80]
        Wandb.log(lg, Dict(
            "loss" => total_loss / length(loader),
            "accuracy" => results.accuracy,           # New!
            "precision_at_10" => results.precision_k,
            "spearman_correlation" => results.spearman,
            "inference_time_ms" => results.time_ms,   # New!
            "epoch" => epoch
        ))
        # -----------------------
        
        if epoch % 10 == 0
             @printf("Epoch %d | Loss: %.4f | Prec@10: %.2f | Time: %.2fms\n", 
                     epoch, total_loss/length(loader), results.precision_k, results.time_ms)
        end
    end
    
    Wandb.close(lg)
    return model
end