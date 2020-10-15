using FeatureNet
using RoamesGeometry

pc = load_pointcloud("testdata/test.las")

# for now just pulling AGL out of Z
pc = map(v -> merge(v,(agl = v.position[3],)),pc)

radii = [5,7,10]

features, classification = generate_features(pc, radii)
success = save_features(features, classification, "testdata/test_duplicated.jls", 10000)


data, classification = load_features("testdata/test_duplicated.jls")

for i = 1:3
    println(data[i,:])
    println("..")
end

features, classification = generate_features(pc, radii)
success = save_features(features, classification, "testdata/test.jls", nothing) 

data, classification = load_features("testdata/test.jls")

for i = 1:3
    println(data[i,:])
    println("..")
end

# 342.0f0, 0.07416348f0, 0.9256414f0, 0.00019507436f0, 0.015892532f0, 0.6933682f0, 10.243818f0, 0.07930529f0, 342.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 342.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, -0.208f0, -0.041f0, 0.232f0, 0.352f0

# 693.0f0, 0.055063605f0, 0.9442368f0, 0.00069960154f0, 0.0022136674f0, 0.6957084f0, 10.112666f0, 0.16699253f0, 691.0f0, 1.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 690.0f0, 0.0f0, 3.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, -0.294f0, -0.032f0, 0.184f0, 1.795f0

# 180.0f0, 0.10534164f0, 0.89453506f0, 0.00012331604f0, 0.018993258f0, 0.69224817f0, 10.1839f0, 0.10016936f0, 180.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 180.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, -0.134f0, -0.0285f0, -0.079f0, 0.123f0