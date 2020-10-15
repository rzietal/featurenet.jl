using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch
using Parameters: @with_kw
using MLDatasets


function getdata(batchsize)
    # Loading Dataset	
    #xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    @show size(xtest)
    @show size(ytest)

    @show typeof(xtest)
    @show typeof(ytest)
	
    # Reshape Data in order to flatten each image into a linear array
    #xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    @show size(xtest)

    # One-hot-encode the labels
    ytest = onehotbatch(ytest, 0:9)

    @show size(ytest)

    # Batching
    #train_data = DataLoader(xtrain, ytrain, batchsize=batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest, batchsize=batchsize)

    return test_data
end

test_data = getdata(10)

#size(xtest) = (28, 28, 10000)
#size(ytest) = (10000,)
#size(xtest) = (784, 10000)
#size(ytest) = (10, 10000)
