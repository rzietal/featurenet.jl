function generate_features(pc, radii::Array, weight::Function=x->one(typeof(x)))

    numpoints = length(pc)

    features = []
    # kdtree = []
    # Generate our kdtree on our bayposition (or position field)
    kdtree = KDTree(pc.position)

    @showprogress 1 "Calculating spatial features..." for i = 1:numpoints
        push!(features, generate_point_features(pc, i, radii, kdtree))
    end

    return features, pc.classification
end

function generate_point_features(pc, ii::Int, radii::Array, kdtree::KDTree, weight::Function=x->one(typeof(x)))
    
    result = Dict{Int, SpatialFeature}()
    position = pc.position

    uretnum = collect(1:7)
    return_number = pc.returnnumber
    num_returns = pc.numberofreturns

    intensity = pc.intensity
    agl = pc.agl

    point = position[ii]

    for radius in radii
        # weight(x) -> exp(-x/radius^2) #exponentially decaying 
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
            logintensityμ += w[i] * log(1 + pc.intensity[j])
            i += 1
        end

        if totalweight !== 0
            logintensityμ = logintensityμ / totalweight
        end


        i = 1
        for j in neighboursidx
            logintensityσ² += w[i] * (log(1 + pc.intensity[j]) - logintensityμ)^2
            i += 1
        end

        if totalweight !== 0
            logintensityσ² = logintensityσ² / totalweight
        end
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
        feature.return_number = Int8(return_number[ii])
        feature.num_returns = Int8(num_returns[ii]) 
        
        nn_agl = agl[neighboursidx]
        feature.agl_hist = [minimum(nn_agl),median(nn_agl), agl[ii], maximum(nn_agl)]

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


function load_features(filename::String)

    data = deserialize(filename)
    labels = data["labels"]
    features = data["features"]

    return features, labels

end



function save_features(features, classification, filename::String, num_features::Union{Int, Nothing})

    num_features_total = length(features)
    selected_features = []
    selected_classifications = []

    if !isnothing(num_features)
        @info "Balancing out the labels..."
        features, classification = feature_data_fillings_duplicate(features, classification, num_features)
        @info "Randomizing and selecting $num_features samples..."
        features, classification, counter = feature_randomise_subsample(features, classification, num_features)
    end
    dict_keys = sort(collect(keys(features[1])))

    NaNcount = 0
    @showprogress 1 "Reformatting and saving..." for i=1:length(features)
        value = features[i]
        line1, success1 = spatialfeature_to_array(value[dict_keys[1]])
        line2, success2 = spatialfeature_to_array(value[dict_keys[2]])
        line3, success3 = spatialfeature_to_array(value[dict_keys[3]])
 
        if isnothing(num_features) # if no duplication then output all the features, even with NaNs etc...
            for l1 in line1
                push!(selected_features, l1)
            end
            for l2 in line2
                push!(selected_features, l2)
            end
            for l3 in line3
                push!(selected_features, l3)
            end
        else # if duplicating for training skip the NaNs
            if success1 && success2 && success3 
                for l1 in line1
                    push!(selected_features, l1)
                end
                for l2 in line2
                    push!(selected_features, l2)
                end
                for l3 in line3
                    push!(selected_features, l3)
                end
                push!(selected_classifications, classification[i])
            else
                NaNcount = NaNcount + 1
            end
        end
    end

    selected_features = reshape(selected_features, featurelength*3, :)
    selected_features = convert(Array, transpose(selected_features))
    data = Dict{String, Any}()
    data["features"] = selected_features

    if !isnothing(num_features)
        classification = selected_classifications
    end
    
    data["labels"] = classification

    @info "Done selecting $(length(classification)) points."
    if NaNcount == 0
        @info "Encountered $(NaNcount) invalid values."
    else
        @warn "Encountered $(NaNcount) invalid values."
    end

    serialize(filename, data)

    return isfile(filename)

end


function feature_data_fillings_duplicate(features::Array, labels::Array, sample_limit::Int)
    num_of_points = length(labels)
    indices_per_label = group(labels, 1:num_of_points)

    sample_limit = round(Int, sample_limit)

    new_indices_per_label = map(i -> repeat_collect(i, sample_limit), collect(indices_per_label))
    
    new_indices = collect(Iterators.flatten(new_indices_per_label))

    return features[new_indices], labels[new_indices]    
end

function repeat_collect(array, min_size)
    if (length(array) >= min_size) || (length(array) == 0)
        return array
    else
        # some iterator magic to repeat the array infinitely, then take the wanted amount
        iterator = Iterators.flatten(Iterators.repeated(array, typemax(Int)))
        return collect(Iterators.take(iterator, min_size))
    end
end

function feature_randomise_subsample(features::Array, classification::Array, sample_limit::Int)

    # Randomise the point cloud so the subset of training samples 
    # will not be from the same cluster/spot/region of the point cloud
    num_of_points = length(classification)

    # group indices by label:
    rng = MersenneTwister(num_of_points)
    indices_per_label = collect(group(classification, 1:num_of_points))
    
    # take shuffled subsets:
    indices_per_label = map(i -> shuffle(rng, i), indices_per_label)
    subsets = map(i -> i[1:min(sample_limit, length(i))], indices_per_label)

    # and merge all those indices:
    point_indices = collect(Iterators.flatten(subsets))

    # select that data:
    new_labels = classification[point_indices]
    new_features = features[point_indices]

    # count and return as Dict{Any,Any}
    counts = groupcount(new_labels)
    unique_labels = collect(keys(counts))
    counter = Dict{Any,Any}( map(l -> l => counts[l], unique_labels))

    return new_features, new_labels, counter
end