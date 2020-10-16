using RoamesGeometry
using FeatureNet

pointcloud = "/home/rzietal/git/featurenet.jl/testdata/test.las"

classified_pointcloud = "/home/rzietal/git/featurenet.jl/testdata/classified_test.las"
model = "/home/rzietal/git/featurenet.jl/testdata/models/model_epoch_10_accuracy_.0.9055.jls"
filename = "/home/rzietal/git/featurenet.jl/testdata/test.jls"

labels = classify(filename, model)

pc = load_pointcloud(pointcloud)

pc.classification .= labels
pc.returnnumber .= 1

save_pointcloud(classified_pointcloud, pc)