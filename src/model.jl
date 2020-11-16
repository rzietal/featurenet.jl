
featureMLP(nfeatures, ) = 
    Chain(
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, selu),
            Dense(nfeatures, nfeatures),
            BatchNorm(nfeatures, selu),
            Dense(nfeatures, nfeatures)
        )
    
struct featureMaxPool
end

struct FeatureNet
    featureMLP
    featureMaxPool

end

@functor FeatureNet
