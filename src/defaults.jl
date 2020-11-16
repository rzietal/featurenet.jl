const SMALLEST_EIGENVALUE = 1e-12
const featurelength = 28
const nclasses = 10

# Feature struct for points - 28 features for single radius
mutable struct SpatialFeature
    num_points          ::Int64
    linearity           ::Float64
    planarity           ::Float64
    sphericity          ::Float64
    verticality         ::Float64
    entropy             ::Float64
    log_intensity_μ     ::Float64
    log_intensity_σ²    ::Float64
    return_number       ::Int8
    num_returns         ::Int8
    return_number_hist  ::SVector{7, Int64}
    num_returns_hist    ::SVector{7, Int64}
    agl_hist            ::SVector{4, Float32}
end

function Base.zero(::Type{SpatialFeature})
    n = Int64(0)
    N7 = SVector(n,n,n,n,n,n,n)
    n = Float64(NaN)
    N4 = SVector(n,n,n,n)
    SpatialFeature(Int64(0), n, n, n, n, n, n, n, Int8(0), Int8(0), N7, N7, N4)
end

function spatialfeature_to_array(feature)

    f = [feature.num_points, feature.linearity, feature.planarity, feature.sphericity, 
         feature.verticality, feature.entropy, feature.log_intensity_μ, feature.log_intensity_σ²,
         feature.return_number, feature.num_returns] 

    for x in feature.return_number_hist
        push!(f,x)
    end
    for x in feature.num_returns_hist
        push!(f,x)
    end
    for x in feature.agl_hist
        push!(f,x)
    end

    if maximum(isnan.(f)) == 1
        return f
    else
        return f
    end
end