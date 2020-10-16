using FeatureNet
using CUDA
using Flux
using Flux.Data: DataLoader

train_dir = "/home/rzietal/git/featurenet.jl/testdata/training"
test_dir = "/home/rzietal/git/featurenet.jl/testdata/testing"

model_dir = "/home/rzietal/git/featurenet.jl/testdata/models/"

train(train_dir, test_dir, 100, 1, 1000, 0.0001, model_dir)