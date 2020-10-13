const SMALLEST_EIGENVALUE = 1e-12

# Feature struct for points
mutable struct SpatialFeature
    num_points          ::Int64
    linearity           ::Float32
    planarity           ::Float32
    sphericity          ::Float32
    verticality         ::Float32
    entropy             ::Float32
    log_intensity_μ     ::Float32
    log_intensity_σ²    ::Float32
    return_number_hist  ::SVector{7, Int64}
    num_returns_hist    ::SVector{7, Int64}
    agl_hist            ::SVector{4, Float32}
end

function Base.zero(::Type{SpatialFeature})
    n = Int64(0)
    N7 = SVector(n,n,n,n,n,n,n)
    n = Float32(NaN)
    N4 = SVector(n,n,n,n)
    SpatialFeature(Int64(0), n, n, n, n, n, n, n, N7, N7, N4)
end