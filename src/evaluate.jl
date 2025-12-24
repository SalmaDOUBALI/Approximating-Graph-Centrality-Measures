using StatsBase, Dates

"""
    evaluate_performance(model, X, Y_true, bc_true, k)
Calculates all project metrics: Precision@K, Spearman, Accuracy, and Inference Time.
"""
function evaluate_performance(model, X, Y_true, bc_true, k)
    # --- 1. Measure Inference Time ---
    # Measuring the time it takes for the lightweight MLP to predict [cite: 17, 44]
    start_time = time_ns()
    raw_output = model(X)
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
    # Focus on the top-k most influential nodes [cite: 18, 46]
    top_k_pred_idx = sortperm(probs, rev=true)[1:k]
    true_top_k_idx = sortperm(bc_true, rev=true)[1:k]
    intersection = intersect(top_k_pred_idx, true_top_k_idx)
    precision_k = length(intersection) / k

    # --- 4. Spearman Rank Correlation ---
    # Measures the quality of the ranking approximation [cite: 67, 80]
    spearman_corr = corspearman(probs, bc_true)

    return (
        accuracy = accuracy,
        precision_k = precision_k,
        spearman = spearman_corr,
        time_ms = inference_time_ms
    )
end