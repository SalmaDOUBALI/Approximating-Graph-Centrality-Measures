using JLD2, FileIO # Required for saving/loading
include("../src/data_generation.jl")
include("../src/model_MLP.jl")
include("../src/evaluate.jl")
include("../src/train.jl")

# Setup Configuration
config = Dict(
    "run_name"    => "BA-2000nodes-100graphs",
    "graph_type"  => "BA",
    "n_graphs"    => 100,
    "n_nodes"     => 2000,
    "top_k"       => 10,     # Focus on top-k nodes [cite: 18, 43]
    "hidden_size" => 32,
    "batch_size"  => 256,
    "lr"          => 0.005,
    "epochs"      => 50
)

# --- 1. Load or Generate Data ---
# Define the path where the data should live
data_dir = "data"
data_filename = "$(config["graph_type"])_$(config["n_nodes"])n_$(config["n_graphs"])g.jld2"
data_path = joinpath(data_dir, data_filename)

# Create data folder if it doesn't exist
if !ispath(data_dir)
    mkpath(data_dir)
end

local dataset
if isfile(data_path)
    println("--- Loading existing dataset from $data_path ---")
    dataset = load(data_path, "dataset")
else
    println("--- Dataset not found. Generating new data... ---")
    # Using node degree information to prepare training sets [cite: 45]
    dataset = generate_graph_dataset(
        config["graph_type"], 
        config["n_graphs"], 
        config["n_nodes"], 
        config["top_k"]
    )
    
    println("--- Saving dataset to $data_path ---")
    save(data_path, "dataset", dataset)
end

# --- 2. Split (80% Train / 20% Test) ---
train_data = dataset[1:80]
test_data  = dataset[81:100]

# --- 3. Train ---
# Lightweight MLP to approximate top-k central nodes [cite: 17, 44, 53]
trained_model = train_model(config, train_data, test_data)