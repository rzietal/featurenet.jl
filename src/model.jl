
radiusMLP(nfeatures = featurelength) = 
    Chain(
            Dense(nfeatures, nfeatures, relu),
            BatchNorm(nfeatures),
            Dense(nfeatures, nfeatures, relu),
            BatchNorm(nfeatures),
            Dense(nfeatures, nfeatures, relu),
            BatchNorm(nfeatures),
            Dense(nfeatures, nfeatures, relu),
            BatchNorm(nfeatures),
            Dense(nfeatures, nfeatures, relu),
            BatchNorm(nfeatures),
            
        )

reduceMLP(nfeatures = featurelength, nclasses = nclasses) = 
    Chain(
            Dense(3*nfeatures, 3*nfeatures, relu),
            BatchNorm(3*nfeatures),
            Dense(3*nfeatures, 3*nfeatures, relu),
            BatchNorm(3*nfeatures),
            Dense(3*nfeatures, 2*nfeatures, relu),
            BatchNorm(2*nfeatures),
            Dense(2*nfeatures, nfeatures, relu),
            BatchNorm(nfeatures),
            Dense(nfeatures, nclasses, relu),
            BatchNorm(nclasses),
        )


struct Featurenet
    radiusMLP
    reduceMLP
end

@functor Featurenet

function Featurenet(featurelength::Int = featurelength, nclasses::Int = nclasses)
    Featurenet(radiusMLP(featurelength), reduceMLP(featurelength, nclasses))
end

function (u::Featurenet)(x::AbstractArray{T}) where T

    x1 = x[1:featurelength,:]
    x2 = x[1+featurelength:2*featurelength,:]
    x3 = x[1+2*featurelength:3*featurelength,:]

    l1  = u.radiusMLP(x1)
    l2  = u.radiusMLP(x2)
    l3  = u.radiusMLP(x3)

    L1 = cat(l1,l2,l3;dims=1)
    probabilities = u.reduceMLP(L1)

    return probabilities
end