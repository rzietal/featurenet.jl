
@with_kw mutable struct Dataset

    name::String = ""
    data_directory::String = "/train"
    data_files::Dict{Int,String} = Dict{Int,String}()
    data_files_keys::Array = []
    num_files::Int = 0

    ini :: Function = initialize_dataset

end

function initialize_dataset(name; data_dir = data_dir::String)

    dataset = Dataset()

    dataset.name = name
    dataset.data_directory = data_dir

    data_files = readdir(dataset.data_directory)
    dataset.num_files = length(data_files)
    
    i = 1
    for f in data_files
        dataset.data_files[i] = joinpath(dataset.data_directory, f)
        i = i + 1
    end

    dataset.data_files_keys = sort(collect(keys(dataset.data_files)))

    return dataset
end

function grab_random_files(dataset::Dataset, num_files::Int; drop_processed = true)

    idx = unique(sample(dataset.data_files_keys, min(dataset.num_files, num_files)))
    files = []

    for i in idx
        push!(files, dataset.data_files[i])
        if drop_processed
            pop!(dataset.data_files,i)
        end
    end

    dataset.num_files = length(dataset.data_files)
    dataset.data_files_keys = collect(keys(dataset.data_files))

    return files

end

function load_files(files::Array)

    data, labels = load_features(files[1])

    for file in files[2:end]
        d, l = load_features(file)
        data = vcat(data, d)
        labels = vcat(labels, l)
    end

    data = data'
    data = convert(Array{}, data)

    return data, labels

end