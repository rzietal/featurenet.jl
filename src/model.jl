cleanupMLP(nfeatures = 68, nclasses = 9) = Chain(

        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nfeatures, relu),
        BatchNorm(nfeatures),
        #Dropout(0.5),
        Dense(nfeatures, nclasses, relu),
        BatchNorm(nclasses),
    )


struct Featurenet
    cleanupMLP
end

@functor Featurenet

function Featurenet(featurelength::Int = 68, nclasses::Int = 9)
    Featurenet(cleanupMLP(featurelength, nclasses))
end

function (u::Featurenet)(x::AbstractArray{T}) where T

    probabilities = u.cleanupMLP(x)

    return probabilities
end