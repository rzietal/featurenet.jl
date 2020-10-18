using FeatureNet
using CUDA
using Flux
using Flux.Data: DataLoader

# train_dir = "/home/rzietal/git/featurenet.jl/testdata/training/"
# test_dir = "/home/rzietal/git/featurenet.jl/testdata/testing/"
# model_dir = "/home/rzietal/git/featurenet.jl/testdata/models/"

train_dir = "D:\\Projects\\featurenet.jl\\testdata\\training\\"
test_dir = "D:\\Projects\\featurenet.jl\\testdata\\testing\\"
model_dir = "D:\\Projects\\featurenet.jl\\testdata\\models\\"

nepochs = 250
numfiles = 1
batchsize = 1000
lr = 0.005
lr_drop_rate = 0.95
lr_step = 10
train(train_dir, test_dir, nepochs, numfiles, batchsize, lr, lr_drop_rate, lr_step, model_dir)