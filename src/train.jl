function build_model(; nfeatures=featurelength*3, nclasses=nclasses, p = 0.5)
    return Chain(
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, relu),
            Dropout(p),
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, relu),
            Dropout(p),
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, relu),
            Dropout(p),
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, relu),
            Dropout(p),
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, relu),
            Dropout(p),
            Dense(nfeatures, 56),
            BatchNorm(56, relu),
            Dense(56, 28),
            BatchNorm(28, relu),
            Dense(28, nclasses),
            BatchNorm(nclasses),
            softmax
            )
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

function loss_all_weighted(dataloader, model, weights)
    l = 0f0
    for (x,y) in dataloader
        l += weighted_logitcrossentropy(model(x),y; weights = weights)
    end
    l/length(dataloader)
end

function weighted_logitcrossentropy(x,y; weights = ones(nclasses), ϵ = eps(float(eltype(x))))
    mean(-sum( y .* log.(x .+ ϵ) .* weights; dims=1))
end

function get_weights(y)

    counts = length.(group(Int.(y)))
    weights = ones(nclasses)
    for i=0:nclasses-1
        weights[i+1] = counts[i]
    end
    weights = maximum(weights) .- weights
    weights = 1000 * weights/maximum(weights) .+ 1

    return weights
end

function train(train_dir, test_dir, nepochs, numfiles, batchsize, lr, lr_drop_rate, lr_step, model_dir)
    
    #initialize datasets
    train_dataset = initialize_dataset("train"; data_dir = train_dir)
    test_dataset = initialize_dataset("test"; data_dir = test_dir)

    # Construct model
    m = build_model()
    m = gpu(m)

    # Define loss
    #loss(x,y) = logitcrossentropy(m(x), y)
    

    # Load testing data 
    test_batch = grab_random_files(test_dataset, 1; drop_processed = false)
    xtest, ytest = load_files(test_batch)
    xtest = convert(Array{Float64}, xtest)
    ytest = onehotbatch(ytest, 0:nclasses-1)
    ytest = convert(Array{Int8}, ytest)

    test_data = DataLoader(xtest |> gpu, ytest |> gpu, batchsize=batchsize) |> gpu

    # Load training data 
    train_batch = grab_random_files(train_dataset, numfiles)
    xtrain, ytrain = load_files(train_batch)

    # get weights before turning into one hot
    weights = get_weights(ytrain)

    xtrain = convert(Array{Float64}, xtrain)

    ytrain = onehotbatch(ytrain, 0:nclasses-1)
    ytrain = convert(Array{Int8}, ytrain)

    train_data = DataLoader(xtrain |> gpu, ytrain |> gpu, batchsize=batchsize, shuffle=true) |> gpu

    # define loss
    loss(x,y) = weighted_logitcrossentropy(m(x),y; weights = weights)

    evalcb = () -> @show(loss_all_weighted(test_data, m, weights))
    opt = ADAM(lr)

    for i = 1:nepochs
        if i % lr_step == 0
            opt.eta = maximum([1e-6, opt.eta*lr_drop_rate])
            @info "New learning rate $(opt.eta)"
        end
        @info "Training epoch $(i)"
        Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

        test_accuracy = accuracy(test_data, m)
        #print out accuracies
        @info "Accuracy on a testing set $(test_accuracy)"

        acc = string(test_accuracy)
        acc = acc[1:min(6,length(acc))]

        serialize(joinpath(model_dir,"model_epoch_$(i)_accuracy_$(acc).jls"), m)
    end

end