module FeatureNet

using FugroGeometry
#using FileIO
#using FugroLAS
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

using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, mse
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDAapi
using CUDA

include("model.jl")
include("train.jl")
include("predict.jl")

export train, classify, Featurenet

end # module