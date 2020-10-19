using RoamesGeometry
using FeatureNet
using Serialization
using CUDA
using Flux
using Flux.Data: DataLoader

testfeatures = "/home/rzietal/git/featurenet.jl/testdata/test_duplicated.jls"
model = "/home/rzietal/git/featurenet.jl/testdata/models/model_epoch_500_MSE_245.2443.jls"

m = deserialize(model) |> cpu

test_data, ~ = load_files([testfeatures])
xtest = convert(Array{Float64}, test_data[1:26,:])
ytest = convert(Array{Float64}, test_data[27:52,:])

for i = 1:5
    a = ytest[:,i]
    b = m(xtest[:,i:i])

    for j = 1:length(a)
        g = a[j]
        p = b[j]
        println("Ground truth: $(g), prediction $(p)")
    end

    println("********************")
end

"Done!"




