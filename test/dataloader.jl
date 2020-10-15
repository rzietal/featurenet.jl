using FeatureNet

train_dir = "/home/rzietal/git/turboboost/train_low/data"
test_dir = "/home/rzietal/git/turboboost/train_low/validation/"

@show train_dataset = initialize_dataset("training"; data_dir = train_dir)
@show test_dataset = initialize_dataset("testing"; data_dir = test_dir)

@show train_batch = grab_random_files(train_dataset, 5)
@show test_batch = grab_random_files(test_dataset, 14; drop_processed = false)

@show train_dataset
@show test_dataset


test_files = ["/home/rzietal/git/featurenet.jl/testdata/test_1.jls", "/home/rzietal/git/featurenet.jl/testdata/test_2.jls"]


data, labels = load_files(test_files)

@show size(data)
@show size(labels)

@info "Done!"