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

function train(data_file, nepochs, batchsize, lr, lr_drop_rate, lr_step, model_dir)
    
    # Construct model
    m = Featurenet()
    m = gpu(m)

    # Define loss
    loss(x,y) = logitcrossentropy(m(x), y)
    

    data = deserialize(data_file)
    labels = data[:,1]
    features = data[:,2:end]

    ids = collect(1:length(labels))
    ids = shuffle(ids)

    labels = labels[ids]
    features = features[ids,:]

    # split 80/20 for training/testing
    @show ntrain = Int(round(0.8*length(labels)))
    @show ntest = length(labels) - ntrain

    # Load testing data 

    xtest = convert(Array{Float64}, features[ntrain+1:end,:]')
    ytest = onehotbatch(labels[ntrain+1:end], 0:8)
    ytest = convert(Array{Int8}, ytest)

    @show size(xtest)
    @show size(ytest)

    test_data = DataLoader(xtest |> gpu, ytest |> gpu, batchsize=batchsize) |> gpu

    xtrain = convert(Array{Float64}, features[1:ntrain,:]')
    ytrain = onehotbatch(labels[1:ntrain], 0:8)
    ytrain = convert(Array{Int8}, ytrain)

    @show size(xtrain)
    @show size(ytrain)

    train_data = DataLoader(xtrain |> gpu, ytrain |> gpu, batchsize=batchsize, shuffle=true) |> gpu

    evalcb = () -> @show(loss_all(test_data, m))
    opt = RMSProp(lr)

    for i = 1:nepochs
        if i % lr_step == 0
            opt.eta = maximum([1e-6, opt.eta*lr_drop_rate])
            @info "New learning rate $(opt.eta)"
        end
        @info "Training epoch $(i)"
        @time Flux.train!(loss, params(m), test_data, opt, cb = evalcb)
        test_accuracy = accuracy(test_data, m)
        #print out accuracies
        @info "Accuracy on a testing set $(test_accuracy)"

        acc = string(test_accuracy)
        acc = acc[1:min(6,length(acc))]

        serialize(joinpath(model_dir,"model_epoch_$(i)_accuracy_$(acc).jls"), m)
    end

end