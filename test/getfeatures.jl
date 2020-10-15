using FeatureNet
using RoamesGeometry

pc = load_pointcloud("testdata/test.las")

# for now just pulling AGL out of Z
pc = map(v -> merge(v,(agl = v.position[3],)),pc)

radii = [5,7,10]

features, classification = generate_features(pc, radii)
success = save_features(features, classification, "testdata/test.jls", 5000)

data, classification = load_features("testdata/test.jls")

for i = 1:3
    println(data[i,:])
    println("..")
end
