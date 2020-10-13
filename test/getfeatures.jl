using FeatureNet
using RoamesGeometry

pc = load_pointcloud("testdata/test.las")

# for now just pulling AGL out of Z
pc = map(v -> merge(v,(agl = v.position[3],)),pc)

features, classification = generate_features(pc, [5,7,10])

success = save_features(features, "testdata/test.jld")