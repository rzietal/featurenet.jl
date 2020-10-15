using FeatureNet

train_dir = "/home/rzietal/git/featurenet.jl/testdata/training"
test_dir = "/home/rzietal/git/featurenet.jl/testdata/testing"

train(train_dir, test_dir, 1, 1, 500, 0.003)