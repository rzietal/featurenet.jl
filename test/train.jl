using FeatureNet

train_dir = "/home/rzietal/git/featurenet.jl/testdata/training"
test_dir = "/home/rzietal/git/featurenet.jl/testdata/testing"

model_dir = "/home/rzietal/git/featurenet.jl/testdata/models/"

train(train_dir, test_dir, 25, 1, 500, 0.0005, model_dir)