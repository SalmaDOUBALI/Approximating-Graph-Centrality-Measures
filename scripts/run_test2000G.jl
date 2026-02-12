using JLD2, FileIO, Random

include("../src/Model_ranking_MLP.jl")
include("../src/evaluate.jl")
include("../src/train_ranking.jl")

config = Dict(
    "run_name"    => "Ranking-MLP-2000g",
    "n_nodes"     => 5000,
    "hidden_size" => 64,      
    "lr"          => 0.0001,  
    "epochs"      => 50,
    "top_k"       => 50       
)

data_path = "data/combined_dataset_2000g_5000n.jld2"

if isfile(data_path)
    println("Loading the dataset : $data_path ...")
    full_dataset = load(data_path, "dataset")
    println("$(length(full_dataset)) graphes loaded sucessfully.")
else
    error("ERROR : file $data_path does not exist. Verify the path")
end

shuffle!(full_dataset)

n_total = length(full_dataset)
n_train = Int(round(0.8 * n_total))

train_data = full_dataset[1:n_train]
test_data  = full_dataset[n_train+1:end]

println("$n_train graphs to train / $(length(test_data)) graphs to test.")

println("Start training...")
trained_model = train_model(config, train_data, test_data)

model_save_path = "data/model_ranking_combined_top1.jld2"
save(model_save_path, "model", trained_model)

println("End")
println("Model saved in : $model_save_path")