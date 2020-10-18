using FeatureNet
using CUDA
using Flux
using Flux.Data: DataLoader

train_dir = "/home/rzietal/git/featurenet.jl/testdata/training"
test_dir = "/home/rzietal/git/featurenet.jl/testdata/testing"

model_dir = "/home/rzietal/git/featurenet.jl/testdata/models/"

#train(train_dir, test_dir, 100, 1, 1000, 0.0001, model_dir)

nepochs = 100
numfiles = 1
batchsize = 1000
lr = 0.01
lr_drop_rate = 0.9
lr_step = 5
train_features(train_dir, test_dir, nepochs, numfiles, batchsize, lr, lr_drop_rate, lr_step, model_dir)