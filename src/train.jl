function build_model(; nfeatures=78, nclasses=5)
    return Chain(
            Dense(nfeatures, 52),
            BatchNorm(52, relu),
            Dense(52, 26),
            BatchNorm(26, relu),
            Dense(26, nclasses))
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

function train(train_dir, test_dir, nepochs, numfiles, batchsize, lr, model_dir)
    
    #initialize datasets
    train_dataset = initialize_dataset("train"; data_dir = train_dir)
    test_dataset = initialize_dataset("test"; data_dir = test_dir)

    # Construct model
    m = build_model()
    m = gpu(m)

    # Define loss
    loss(x,y) = logitcrossentropy(m(x), y)

    # Load testing data 
    test_batch = grab_random_files(test_dataset, 1; drop_processed = false)
    xtest, ytest = load_files(test_batch)
    xtest = convert(Array{Float64}, xtest)
    ytest = onehotbatch(ytest, 0:4)
    ytest = convert(Array{Int8}, ytest)

    test_data = DataLoader(xtest |> gpu, ytest |> gpu, batchsize=batchsize) |> gpu

    evalcb = () -> @show(loss_all(test_data, m))

    opt = ADAM(lr)

    for i = 1:nepochs

        while train_dataset.num_files > 0

            # Load training data 
            train_batch = grab_random_files(train_dataset, numfiles)
            xtrain, ytrain = load_files(train_batch)
            xtrain = convert(Array{Float64}, xtrain)

            ytrain = onehotbatch(ytrain, 0:4)
            ytrain = convert(Array{Int8}, ytrain)

            train_data = DataLoader(xtrain |> gpu, ytrain |> gpu, batchsize=batchsize, shuffle=true) |> gpu

            @info "Training..."
            Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

        end

        #re-initialize dataset after all files processed
        train_dataset = initialize_dataset("train"; data_dir = train_dir)
        test_accuracy = accuracy(test_data, m)
        #print out accuracies
        @info "Epoch $(i)"
        @info "Accuracy on a testing set $(test_accuracy)"

        acc = string(test_accuracy)[1:6]

        serialize(joinpath(model_dir,"model_epoch_$(i)_accuracy_.$(acc).jls"), m)
    end

    

end