k = 4 
num_val_samples = Int(size(train_data)[2]/k)
epochs = 500 
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
    end 
    #Computation of the MAE
    y_hat = model(val_data)
    mae = sum(abs.(y_hat - val_targets))/length(val_targets)
    push!(all_scores, mae)
end 

accuracy = mean(all_scores)