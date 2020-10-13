function generate_features(pc, radii::Array, weight::Function=x->one(typeof(x)))

    numpoints = length(pc)

    features = []
    # kdtree = []
    # Generate our kdtree on our bayposition (or position field)
    kdtree = KDTree(pc.position)

    @showprogress 1 "Calculating spatial features..." for i = 1:numpoints
        push!(features, generate_point_features(pc, i, radii, kdtree::KDTree))
    end

    return features, pc.classification
end

function generate_point_features(pc, i::Int, radii::Array, kdtree::KDTree, weight::Function=x->one(typeof(x)))
    
    result = Dict{Int, SpatialFeature}()
    position = pc.position

    uretnum = collect(1:7)
    return_number = pc.returnnumber
    num_returns = pc.numberofreturns

    intensity = pc.intensity
    agl = pc.agl

    point = position[i]

    for radius in radii
        neighboursidx =  inrange(kdtree, point, radius)
    
        # Set up our features we want to calculate
        mycovariance = zeros(SMatrix{3, 3, Float64})
        w = zeros(length(neighboursidx))
        logintensityμ = zero(Float32)
        logintensityσ² = zero(Float32)

        i = 1
        totalweight = 0
        sx = 1.0 # neighbourhood scaling
        sy = 1.0 # helps with verticality
        sz = 1.0 # turned off here

        for j in neighboursidx
            p = position[j] - point
            sp = Point3(sx*p[1], sy*p[2],sz*p[3])
            w[i] = weight(norm(sp)^2)
            totalweight += w[i]
            mycovariance += w[i] * sp * sp'
            logintensityμ += w[i] * log(pc.intensity[j])
            i += 1
        end

        logintensityμ = logintensityμ / totalweight

        i = 1
        for j in neighboursidx
            logintensityσ² += w[i] * (log(pc.intensity[j]) - logintensityμ)^2
            i += 1
        end

        logintensityσ² = logintensityσ² / totalweight

        # Calculate our local spatial information using PCA
        
        eig = eigen(Matrix(mycovariance))
        values = map(v -> max(v,SMALLEST_EIGENVALUE),real.(eig.values)) # make sure eigenvalues are positive definite
        vectors = real.(eig.vectors)
        principvect = sum(abs.(vectors).*values',dims=2)
        principvect = principvect/norm(principvect)
        verticality = principvect[3]

        sortedindex = sortperm(values, rev = true)
        l1 = values[sortedindex[1]]
        l2 = values[sortedindex[2]]
        l3 = values[sortedindex[3]]
        l = sum(values)
        
        entropy = - sum(x -> (x/l)*log(x/l),values)

        # http://dgl.geomatics.ncku.edu.tw/Papers/Point%20Cloud%20Classification.pdf as defined in section 2.2
        # Interesting geometric median weighting proposed - could look into later probably high-ER compute cost
        linearity = (l1 - l2) / l1
        planarity = (l2 - l3) / l1
        sphericity = l3 / l1

        feature = zero(SpatialFeature)

        feature.num_points = length(neighboursidx)
        feature.linearity = linearity
        feature.planarity = planarity
        feature.sphericity = sphericity
        feature.verticality = verticality
        feature.entropy = entropy
        feature.log_intensity_μ = logintensityμ
        feature.log_intensity_σ² = logintensityσ²

        nn_agl = agl[neighboursidx]
        feature.agl_hist = [minimum(nn_agl),median(nn_agl), agl[i], maximum(nn_agl)]

        nn_returnnumber = return_number[neighboursidx]
        return_number_hist = map(v -> count(x->x==v,nn_returnnumber), uretnum)
        feature.return_number_hist = return_number_hist

        nn_numreturns = num_returns[neighboursidx]
        num_returns_hist = map(v -> count(x->x==v,nn_numreturns), uretnum)
        feature.num_returns_hist = num_returns_hist

        result[radius] = feature
    end

    return result
end

function save_features(features, filename::String)

    num_features = length(features)

    selected_features = Dict{String, Any}()

    for key in keys(features[1])
        selected_features["radius_$(key)"] = Any[]
    end

    @showprogress 1 "Reformatting and saving features..." for i=1:num_features
        for (key, value) in features[i]
            push!(selected_features["radius_$(key)"], value)
        end
    end

    save(filename, selected_features)

    return isfile(filename)

end