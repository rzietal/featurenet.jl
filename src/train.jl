
if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    import CuArrays		# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end

function build_model(; nfeatures=78, nclasses=5)
    return Chain(
            Dense(nfeatures, 26, relu),
            Dense(26, nclasses))
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end

function train(train_dir, test_dir, nepochs, numfiles, batchsize, lr)
    
    #initialize datasets
    train_dataset = initialize_dataset("train"; data_dir = train_dir)
    test_dataset = initialize_dataset("train"; data_dir = test_dir)

    # Construct model
    m = build_model()
    m = gpu(m)

    # Define loss
    loss(x,y) = logitcrossentropy(m(x), y)

    opt = ADAM(lr)

    @showprogress 1 "Training..." for i = 1:nepochs

        while train_dataset.num_files > 0

            # Load testing data 
            test_batch = grab_random_files(test_dataset, 1; drop_processed = false)
            xtest, ytest = load_files(test_batch)
            ytest = onehotbatch(ytest, 0:4)
            test_data = DataLoader(xtest, ytest, batchsize=batchsize)

            # Load training data 
            train_batch = grab_random_files(train_dataset, numfiles)
            xtrain, ytrain = load_files(train_batch)

            ytrain = onehotbatch(ytrain, 0:4)
            train_data = DataLoader(xtrain, ytrain, batchsize=batchsize)
            train_data = gpu.(train_data)
            test_data = gpu.(test_data)

            Flux.train!(loss, params(m), train_data, opt)

            @show accuracy(train_data, m)
            @show accuracy(test_data, m)
        end

        #re-initialize dataset after all files processed
        train_dataset = initialize_dataset("train"; data_dir = train_dir)
    end

end