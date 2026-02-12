using StatsBase, Dates

function evaluate_performance(model, X, Y_true, bc_true, k)
    start_time = time_ns()
    raw_output = model(X)
    probs = softmax(raw_output)[1, :]
    end_time = time_ns()
    
    inference_time_ms = (end_time - start_time) / 1e6

    # Classification Accuracy:
    accuracy = 0.0
    if Y_true !== nothing
        predictions = probs .> 0.5
        true_labels = Y_true[1, :]
        accuracy = mean(predictions .== true_labels)
    end

    # Precision@K:
    top_k_pred_idx = sortperm(probs, rev=true)[1:k]
    true_top_k_idx = sortperm(bc_true, rev=true)[1:k]
    intersection = intersect(top_k_pred_idx, true_top_k_idx)
    precision_k = length(intersection) / k

    # Spearman correlation 
    spearman_corr = corspearman(probs, bc_true)
    
    # Kendall's Tau 
    kendall_corr = corkendall(probs, bc_true)

    return (
        accuracy = accuracy,
        precision_k = precision_k,
        spearman = spearman_corr,
        kendall = kendall_corr,      
        time_ms = inference_time_ms
    )
end