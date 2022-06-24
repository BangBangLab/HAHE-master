vocab_size = {'jf17k':29148, 'wikipeople-':35005, 'wd50k':47688}
num_relations = {'jf17k':501, 'wikipeople-':178, 'wd50k':531}
max_seq_len = {'jf17k':11, 'wikipeople-':13, 'wd50k':29}
max_arity = {'jf17k':6, 'wikipeople-':7, 'wd50k':15}


def process_dataset(dataset_name, config):
    dataset_name = dataset_name.lower()
    if dataset_name == 'wikipeople':
        dataset_name += '-'
    if dataset_name == 'jf17k':
        config["train_file"] = "./data/" + dataset_name + "/train.json"
    else:
        config["train_file"] = "./data/" + dataset_name + "/train+valid.json"
    
    config["predict_file"] = "./data/" + dataset_name + "/test.json"
    config["ground_truth_path"] = "./data/" + dataset_name + "/all.json"
    config["vocab_path"] = "./data/" + dataset_name + "/vocab.txt"
    config["vocab_size"] = vocab_size[dataset_name]
    config["num_relations"] = num_relations[dataset_name]
    config["max_seq_len"] = max_seq_len[dataset_name]
    config["max_arity"] = max_arity[dataset_name]

    return config