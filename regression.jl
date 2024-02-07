using Flux 
using MLDatasets: BostonHousing
using DataFrames
using Statistics
using Flux: train!, params
using Plots

features = BostonHousing.features()
targets = BostonHousing.targets()

train_data = features[:,1:404]
train_targets = targets[:,1:404]

test_data = features[:,405:506]
test_targets = targets[:,405:506]

#We can't feed these numbers (different scales) into a neural network. Normalize the data: 

avg = mean(train_data, dims = 2)
st = std(train_data, dims = 2)

train_data = (train_data .-avg)./st
test_data = (test_data .-avg)./st
#Setup the model: 

function build_model() 
    model = Chain(Dense(13, 64, relu), Dense(64, 64, relu), Dense(64, 1))
    ps = params(model)
    opt = Flux.RMSProp()
    return ps, opt, model
end 

#Since we have too little data, we use K-fold validation rather than using a train and validation set 

k = 4 
num_val_samples = Int(size(train_data)[2]/k)
epochs = 220 
mae_history = zeros(k, epochs)
all_scores = []
val_data = train_data[:,1:num_val_samples]

val_data = train_data[:,(2-1)*num_val_samples : (2)* num_val_samples]
for g in 1:k
    println("Processing fold : $g")
    
    if g == 1 
        val_data = train_data[:,1:num_val_samples]
        val_targets = train_targets[:,1:num_val_samples]
    else 
        val_data = train_data[:,(g-1)* num_val_samples+1: (g)* num_val_samples]
        val_targets = train_targets[:,(g-1)* num_val_samples+1 : (g)* num_val_samples]
    end 

    if g == 1 
        partial_train_data = train_data[:,(g) * num_val_samples + 1:end]
        partial_train_targets = train_targets[:,(g) * num_val_samples + 1:end]
    else 
        partial_train_data = hcat(train_data[:,1:(g-1) * num_val_samples], train_data[:,(g) * num_val_samples + 1:end])
        partial_train_targets = hcat(train_targets[:,1:(g-1) * num_val_samples], train_targets[:,(g) * num_val_samples + 1:end])
    end 

    ps, opt, model = build_model()
    loss(x, y) = Flux.Losses.mse(model(x), y)

    for epoch in 1:epochs 
        train!(loss, ps, [(partial_train_data, partial_train_targets)], opt)
        train_loss = loss(train_data, train_targets)
        println("Epoch = $epoch Training Loss = $train_loss")
        y_hat = model(val_data)
        mae = sum(abs.(y_hat - val_targets))/length(val_targets)
        mae_history[g,epoch] = mae
    end 
    #Computation of the MAE
    y_hat = model(val_data)
    mae = sum(abs.(y_hat - val_targets))/length(val_targets)
    push!(all_scores, mae)
end 

accuracy = mean(all_scores) 
mae_series = transpose(mean(mae_history, dims=1))

plot(mae_series)