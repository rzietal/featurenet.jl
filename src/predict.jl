function classify(filename::String, model::String)

    m = deserialize(model) #|> gpu

    @show typeof(m)

    x, y = load_files([filename])
    x = convert(Array{Float64}, x) #|> gpu
    
    @show typeof(x)
    
    testmode!(m)
    probs = softmax(m(x))
    #cpu(probs)

    maxprob, cartindx = dropdims.(findmax(probs; dims = 1); dims = 1)
    classification = UInt8.(map(v -> v[1]-1, cartindx))

    @show unique(classification)

    return classification
end