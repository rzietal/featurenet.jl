
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

reduce2MLP(nfeatures = featurelength, nclasses = nclasses) = 
        Chain(
                Dense(nfeatures, nfeatures, relu),
                BatchNorm(nfeatures),
                Dense(nfeatures, nfeatures, relu),
                BatchNorm(nfeatures),
                Dense(nfeatures, nfeatures, relu),
                BatchNorm(nfeatures),
                Dense(nfeatures, nfeatures, relu),
                BatchNorm(nfeatures),
                Dense(nfeatures, nclasses, relu),
                BatchNorm(nclasses),   
            )


struct Featurenet
    radiusMLP
    reduceMLP
    reduce2MLP
end

@functor Featurenet

function Featurenet(featurelength::Int = featurelength, nclasses::Int = nclasses)
    Featurenet(radiusMLP(featurelength), reduceMLP(featurelength, nclasses), reduce2MLP(featurelength, nclasses))
end

function (u::Featurenet)(x::AbstractArray{T}) where T

    x1 = x[1:featurelength,:]
    x2 = x[1+featurelength:2*featurelength,:]
    x3 = x[1+2*featurelength:3*featurelength,:]

    l1  = u.radiusMLP(x1)
    l2  = u.radiusMLP(x2)
    l3  = u.radiusMLP(x3)

    L1 = cat(l1,l2,l3;dims=1)

    L2 = dropdims(maximum(cat(l1,l2,l3;dims=3);dims=3);dims=3)

    p1 = u.reduceMLP(L1)
    p2 = u.reduce2MLP(L2)

    probabilities = dropdims(maximum(cat(p1,p2;dims=3);dims=3);dims=3)

    return probabilities
end