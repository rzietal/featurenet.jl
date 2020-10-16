function classify(filename::String, model::String)

    m = deserialize(model)
    m = gpu(m)

    x, y = load_files([filename])

    @show size(x)

    probs = softmax(m(x))

    maxprob, cartindx = dropdims.(findmax(probs; dims = 1); dims = 1)
    classification = UInt8.(map(v -> v[1]-1, cartindx))

    @show unique(classification)

    return classification
end