using FeatureNet

nepochs = 100
batchsize = 10000
lr = 0.001
lr_drop_rate = 0.95
lr_step = 2
model_dir = "testdata/"
train("testdata/colac_AB_urban.featurenet.jls", nepochs, batchsize, lr, lr_drop_rate, lr_step, model_dir)