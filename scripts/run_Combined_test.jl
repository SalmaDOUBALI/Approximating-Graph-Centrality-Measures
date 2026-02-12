using JLD2, FileIO, Random
include("../src/data_generation.jl")
include("../src/Model_MLP.jl")
include("../src/evaluate.jl")
include("../src/train.jl")

config = Dict(
    "run_name"    => "Combined-4Types-5000n2000g",
    "n_nodes"     => 5000,
    "graphs_per_type" => 500, 
    "hidden_size" => 64,      
    "batch_size"  => 512,     
    "lr"          => 0.0005,  
    "epochs"      => 50       
)

data_path = "data/combined_dataset_2000g_5000n.jld2"
full_dataset = []

if isfile(data_path)
    println("loading the existant dataset...")
    full_dataset = load(data_path, "dataset")
else
    graph_types = ["ER", "BA", "NWS", "SSF"]
    # Top 1% 
    k_1_percent = 50 
    
    for type in graph_types
        println("\n--- Generating type: $type ---")
        sub_dataset = generate_graph_dataset(type, config["graphs_per_type"], config["n_nodes"], k_1_percent)
        append!(full_dataset, sub_dataset)

        save("data/checkpoint_$(type).jld2", "dataset", sub_dataset)
    end
    
    println("dataset shuffling...")
    shuffle!(full_dataset)
    
    println("Saving the full dataset...")
    save(data_path, "dataset", full_dataset)
end


train_data = full_dataset[1:1600]
test_data  = full_dataset[1601:2000]

println("Start training on 2000 graphs...")
trained_model = train_model(config, train_data, test_data)

save("data/model_combined_top1.jld2", "model", trained_model)
println("End")