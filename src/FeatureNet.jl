module FeatureNet

using RoamesGeometry
using Serialization
using NearestNeighbors
using SplitApplyCombine
using StaticArrays
using Statistics
using Random
using LinearAlgebra
using StatsBase
using TypedTables
using ProgressMeter
using StatsBase
using Serialization

using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDAapi
using CUDA

include("defaults.jl")
include("getfeatures.jl")
include("dataloader.jl")
include("train.jl")
include("predict.jl")

export generate_features, save_features, load_features, initialize_dataset, grab_random_files, load_files, train, classify

end # module