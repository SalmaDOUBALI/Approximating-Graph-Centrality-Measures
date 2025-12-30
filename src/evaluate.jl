using StatsBase, Dates

"""
    evaluate_performance(model, X, Y_true, bc_true, k)
Calculates all project metrics: Precision@K, Spearman, Kendall's Tau, Accuracy, and Inference Time.
"""
function evaluate_performance(model, X, Y_true, bc_true, k)
    # --- 1. Measure Inference Time ---
    # Measuring the time it takes for the lightweight MLP to predict
    start_time = time_ns()
    raw_output = model(X)
    # Applying softmax to get probabilities for the "important" class
    probs = softmax(raw_output)[1, :]
    end_time = time_ns()
    
    # Inference time in milliseconds
    inference_time_ms = (end_time - start_time) / 1e6

    # --- 2. Classification Accuracy ---
    # Standard accuracy: comparison of predicted binary labels vs ground truth 
    predictions = probs .> 0.5
    true_labels = Y_true[1, :]
    accuracy = mean(predictions .== true_labels)

    # --- 3. Precision@K ---
    # Focus on the top-k most influential nodes
    top_k_pred_idx = sortperm(probs, rev=true)[1:k]
    true_top_k_idx = sortperm(bc_true, rev=true)[1:k]
    intersection = intersect(top_k_pred_idx, true_top_k_idx)
    precision_k = length(intersection) / k

    # --- 4. Rank Correlations ---
    # Spearman measures the correlation between ranks
    spearman_corr = corspearman(probs, bc_true)
    
    # Kendall's Tau measures the strength of association between two ranked variables
    # This is excellent for checking if the model preserves the order of importance
    kendall_corr = corkendall(probs, bc_true)

    return (
        accuracy = accuracy,
        precision_k = precision_k,
        spearman = spearman_corr,
        kendall = kendall_corr,      # New Metric!
        time_ms = inference_time_ms
    )
end