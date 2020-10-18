function build_model_features(; nfeatures=26)
    return Chain(
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, relu),
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, relu),
            Dense(nfeatures, nfeatures))
end

function loss_all_features(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += mse(model(x), y)
    end
    l/length(dataloader)
end

function train_features(train_dir, test_dir, nepochs, numfiles, batchsize, lr, lr_drop_rate, lr_step, model_dir)
    
    #initialize datasets
    train_dataset = initialize_dataset("train"; data_dir = train_dir)
    test_dataset = initialize_dataset("test"; data_dir = test_dir)

    # Construct model
    m = build_model_features()
    m = gpu(m)

    # Define loss
    loss(x,y) = mse(m(x), y)

    # Load testing data 
    test_batch = grab_random_files(test_dataset, 1; drop_processed = false)
    test_data, ~ = load_files(test_batch)
    xtest = convert(Array{Float64}, test_data[1:26,:])
    ytest = convert(Array{Float64}, test_data[27:52,:])

    test_data = DataLoader(xtest |> gpu, ytest |> gpu, batchsize=batchsize) |> gpu

    evalcb = () -> @show(loss_all_features(test_data, m))

    opt = ADAM(lr)

    for i = 1:nepochs
        if i % lr_step == 0
            opt.eta = maximum([1e-6, opt.eta*lr_drop_rate])
            @info "New learning rate $(opt.eta)"
        end
        while train_dataset.num_files > 0

            # Load training data 
            train_batch = grab_random_files(train_dataset, numfiles)
            train_data, ~ = load_files(train_batch)
            xtrain = convert(Array{Float64}, train_data[1:26,:])
            ytrain = convert(Array{Float64}, train_data[27:52,:])

            train_data = DataLoader(xtrain |> gpu, ytrain |> gpu, batchsize=batchsize, shuffle=true) |> gpu

            @info "Training..."
            Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

        end

        #re-initialize dataset after all files processed
        train_dataset = initialize_dataset("train"; data_dir = train_dir)
        test_accuracy = loss_all_features(test_data, m)
        #print out accuracies
        @info "Epoch $(i)"
        @info "MSE on a testing set $(test_accuracy)"

        acc = string(test_accuracy)[1:8]

        serialize(joinpath(model_dir,"model_epoch_$(i)_MSE_$(acc).jls"), m)
    end

    

end