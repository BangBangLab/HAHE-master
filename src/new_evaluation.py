import json
import collections
import numpy as np

from reader.data_reader import read_examples

def generate_ground_truth(ground_truth_path, vocabulary, max_arity,
                          max_seq_length):
    """
    Generate ground truth for filtered evaluation.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, " \
        "and max_aux attribute-value pairs."

    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    all_examples, _ = read_examples(ground_truth_path)
    for (example_id, example) in enumerate(all_examples):
        # get padded input tokens and ids
        hrt = [ example.head, example.relation,example.tail]
        aux_av = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_av.append(attribute)
                    aux_av.append(value)

        while len(aux_av) < (max_aux*2):
            aux_av.append("[PAD]")
            aux_av.append("[PAD]")
        assert len(aux_av) == (max_aux*2)

        input_tokens = hrt + aux_av
        input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
        assert len(input_tokens) == max_seq_length
        assert len(input_ids) == max_seq_length

        # get target answer for each pos and the corresponding key
        for pos in range(max_seq_length):
            if input_ids[pos] == 0:
                continue
            key = " ".join([
                str(input_ids[x]) for x in range(max_seq_length) if x != pos
            ])
            gt_dict[pos][key].append(input_ids[pos])

    return gt_dict

def new_batch_evaluation(global_idx, batch_results, all_features, gt_dict, dataset_max_arity, degree_table):
    """
    Perform batch evaluation.
    """
    ret_ranks = {
        'entity': [],
        'relation': [],
        # '2-r': [],
        # '2-ht': [],
        # 'n-r': [],
        # 'n-ht': [],
        # 'n-a': [],
        # 'n-v': []
        'r': [[] for i in range(dataset_max_arity+1)],
        'ht': [[] for i in range(dataset_max_arity+1)],
        'a': [[] for i in range(dataset_max_arity+1)],
        'v': [[] for i in range(dataset_max_arity+1)],
    }

    _deg_ranks = [[] for i in range(0, np.amax(degree_table)+1)]

    for i, result in enumerate(batch_results):
        feature = all_features[global_idx + i]
        feature_arity = min(feature.arity, dataset_max_arity)
        target = feature.mask_label
        pos = feature.mask_position
        key = " ".join([
            str(feature.input_ids[x]) for x in range(len(feature.input_ids))
            if x != pos
        ])

        # filtered setting
        rm_idx = gt_dict[pos][key]
        rm_idx = [x for x in rm_idx if x != target]
        for x in rm_idx:
            result[x] = -np.Inf
        sortidx = np.argsort(result)[::-1]

        if feature.mask_type == 1:
            ret_ranks['entity'].append(np.where(sortidx == target)[0][0] + 1)
            _deg_ranks[degree_table[feature.mask_label]].append(np.where(sortidx == target)[0][0] + 1) 
        elif feature.mask_type == -1:
            ret_ranks['relation'].append(np.where(sortidx == target)[0][0] + 1)
        else:
            raise ValueError("Invalid `feature.mask_type`.")

        # if feature.arity == 2:
        #     if pos == 1:
        #         ret_ranks['2-r'].append(np.where(sortidx == target)[0][0] + 1)
        #     elif pos == 0 or pos == 2:
        #         ret_ranks['2-ht'].append(np.where(sortidx == target)[0][0] + 1)
        #     else:
        #         raise ValueError("Invalid `feature.mask_position`.")
        # elif feature.arity > 2:
        #     if pos == 1:
        #         ret_ranks['n-r'].append(np.where(sortidx == target)[0][0] + 1)
        #     elif pos == 0 or pos == 2:
        #         ret_ranks['n-ht'].append(np.where(sortidx == target)[0][0] + 1)
        #     elif pos > 2 and feature.mask_type == -1:
        #         ret_ranks['n-a'].append(np.where(sortidx == target)[0][0] + 1)
        #     elif pos > 2 and feature.mask_type == 1:
        #         ret_ranks['n-v'].append(np.where(sortidx == target)[0][0] + 1)
        #     else:
        #         raise ValueError("Invalid `feature.mask_position`.")
        # else:
        #     raise ValueError("Invalid `feature.arity`.")
        if pos == 1:
            ret_ranks['r'][feature_arity].append(np.where(sortidx == target)[0][0] + 1)
        elif pos == 0 or pos == 2:
            ret_ranks['ht'][feature_arity].append(np.where(sortidx == target)[0][0] + 1)           
        elif feature.arity > 2:
            if pos > 2 and feature.mask_type == -1:
                ret_ranks['a'][feature_arity].append(np.where(sortidx == target)[0][0] + 1)
            elif pos > 2 and feature.mask_type == 1:
                ret_ranks['v'][feature_arity].append(np.where(sortidx == target)[0][0] + 1)
            else:
                raise ValueError("Invalid `feature.mask_position`.")
        else:
            raise ValueError("Invalid `feature.mask_position`.")


    ent_ranks = np.asarray(ret_ranks['entity'])
    rel_ranks = np.asarray(ret_ranks['relation'])
    _r_ranks = np.asarray(ret_ranks['r'], dtype=object)
    _ht_ranks = np.asarray(ret_ranks['ht'], dtype=object)
    _a_ranks = np.asarray(ret_ranks['a'], dtype=object)
    _v_ranks = np.asarray(ret_ranks['v'], dtype=object)

    return ent_ranks, rel_ranks, \
           _r_ranks, _ht_ranks, _a_ranks, _v_ranks, _deg_ranks


def new_compute_metrics(ent_lst, rel_lst, _r_lst, _ht_lst,
                    _a_lst, _v_lst, _deg_ranks, eval_result_file, dataset_max_arity):
    """
    Combine the ranks from batches into final metrics.
    """
    all_ent_ranks = np.array(ent_lst).ravel()
    all_rel_ranks = np.array(rel_lst).ravel()
    _n_r_ranks = [[] for i in range(2)]
    _n_ht_ranks = [[] for i in range(2)]
    _n_a_ranks = [[] for i in range(2)]
    _n_v_ranks = [[]for i in range(2)]
    all_r_lst = []
    all_ht_lst = []
    for arity in range(2, dataset_max_arity+1):
        _n_r_ranks.extend([np.array(_r_lst[arity]).ravel()])
        all_r_lst.extend(_r_lst[arity])
        _n_ht_ranks.extend([np.array(_ht_lst[arity]).ravel()])
        all_ht_lst.extend(_ht_lst[arity])
        _n_a_ranks.extend([np.array(_a_lst[arity]).ravel()])
        _n_v_ranks.extend([np.array(_v_lst[arity]).ravel()])
    # _r_ranks = np.array([_r_item.ravel() for _r_item in _r_lst])
    # _ht_ranks = np.array(_ht_lst).ravel()
    # _a_ranks = np.array(_a_lst)
    # _2_r_ranks = np.array(_2_r_lst).ravel()
    # _2_ht_ranks = np.array(_2_ht_lst).ravel()
    # _n_r_ranks = np.array(_n_r_lst).ravel()
    # _n_ht_ranks = np.array(_n_ht_lst).ravel()
    # _n_a_ranks = np.array(_n_a_lst).ravel()
    # _n_v_ranks = np.array(_n_v_lst).ravel()
    all_r_ranks = np.array(all_r_lst).ravel()
    all_ht_ranks = np.array(all_ht_lst).ravel()

    mrr_ent = np.mean(1.0 / all_ent_ranks)
    hits1_ent = np.mean(all_ent_ranks <= 1.0)
    hits3_ent = np.mean(all_ent_ranks <= 3.0)
    hits5_ent = np.mean(all_ent_ranks <= 5.0)
    hits10_ent = np.mean(all_ent_ranks <= 10.0)
    num_ent = len(all_ent_ranks)

    mrr_rel = np.mean(1.0 / all_rel_ranks)
    hits1_rel = np.mean(all_rel_ranks <= 1.0)
    hits3_rel = np.mean(all_rel_ranks <= 3.0)
    hits5_rel = np.mean(all_rel_ranks <= 5.0)
    hits10_rel = np.mean(all_rel_ranks <= 10.0)
    num_rel = len(all_rel_ranks)


    mrr_r = np.mean(1.0 / all_r_ranks)
    hits1_r = np.mean(all_r_ranks <= 1.0)
    hits3_r = np.mean(all_r_ranks <= 3.0)
    hits5_r = np.mean(all_r_ranks <= 5.0)
    hits10_r = np.mean(all_r_ranks <= 10.0)
    num_r = len(all_r_ranks)

    mrr_ht = np.mean(1.0 / all_ht_ranks)
    hits1_ht = np.mean(all_ht_ranks <= 1.0)
    hits3_ht = np.mean(all_ht_ranks <= 3.0)
    hits5_ht = np.mean(all_ht_ranks <= 5.0)
    hits10_ht = np.mean(all_ht_ranks <= 10.0)
    num_ht = len(all_ht_ranks)

    mrr_nr = [-1 for i in range(2)]
    hits1_nr = [-1 for i in range(2)]
    hits3_nr = [-1 for i in range(2)]
    hits5_nr = [-1 for i in range(2)] 
    hits10_nr = [-1 for i in range(2)]
    num_nr = [-1 for i in range(2)]

    mrr_nht = [-1 for i in range(2)]
    hits1_nht = [-1 for i in range(2)]
    hits3_nht = [-1 for i in range(2)]
    hits5_nht = [-1 for i in range(2)] 
    hits10_nht = [-1 for i in range(2)]
    num_nht = [-1 for i in range(2)]

    mrr_na = [-1 for i in range(3)]
    hits1_na = [-1 for i in range(3)]
    hits3_na = [-1 for i in range(3)]
    hits5_na = [-1 for i in range(3)] 
    hits10_na = [-1 for i in range(3)]
    num_na = [-1 for i in range(3)]

    mrr_nv = [-1 for i in range(3)]
    hits1_nv = [-1 for i in range(3)]
    hits3_nv = [-1 for i in range(3)]
    hits5_nv = [-1 for i in range(3)] 
    hits10_nv = [-1 for i in range(3)]
    num_nv = [-1 for i in range(3)]

    for arity in range(2, dataset_max_arity+1):
        mrr_nr.append(np.mean(1.0 / _n_r_ranks[arity]))
        hits1_nr.append(np.mean(_n_r_ranks[arity] <= 1.0))
        hits3_nr.append(np.mean(_n_r_ranks[arity] <= 3.0))
        hits5_nr.append(np.mean(_n_r_ranks[arity] <= 5.0))
        hits10_nr.append(np.mean(_n_r_ranks[arity] <= 10.0))
        num_nr.append(len(_n_r_ranks[arity]))

        mrr_nht.append(np.mean(1.0 / _n_ht_ranks[arity]))
        hits1_nht.append(np.mean(_n_ht_ranks[arity] <= 1.0))
        hits3_nht.append(np.mean(_n_ht_ranks[arity] <= 3.0))
        hits5_nht.append(np.mean(_n_ht_ranks[arity] <= 5.0))
        hits10_nht.append(np.mean(_n_ht_ranks[arity] <= 10.0))
        num_nht.append(len(_n_ht_ranks[arity]))

        if arity == 2:
            continue
        else:
            mrr_na.append(np.mean(1.0 / _n_a_ranks[arity]))
            hits1_na.append(np.mean(_n_a_ranks[arity] <= 1.0))
            hits3_na.append(np.mean(_n_a_ranks[arity] <= 3.0))
            hits5_na.append(np.mean(_n_a_ranks[arity] <= 5.0))
            hits10_na.append(np.mean(_n_a_ranks[arity] <= 10.0))
            num_na.append(len(_n_a_ranks[arity]))

            mrr_nv.append(np.mean(1.0 / _n_v_ranks[arity]))
            hits1_nv.append(np.mean(_n_v_ranks[arity] <= 1.0))
            hits3_nv.append(np.mean(_n_v_ranks[arity] <= 3.0))
            hits5_nv.append(np.mean(_n_v_ranks[arity] <= 5.0))
            hits10_nv.append(np.mean(_n_v_ranks[arity] <= 10.0))
            num_nv.append(len(_n_v_ranks[arity]))


    eval_result = {
        'entity': {
            'mrr': mrr_ent,
            'hits1': hits1_ent,
            'hits3': hits3_ent,
            'hits5': hits5_ent,
            'hits10': hits10_ent,
            'num': num_ent
        },
        'relation': {
            'mrr': mrr_rel,
            'hits1': hits1_rel,
            'hits3': hits3_rel,
            'hits5': hits5_rel,
            'hits10': hits10_rel,
            'num': num_rel
        },
        'ht': {
            'mrr': mrr_ht,
            'hits1': hits1_ht,
            'hits3': hits3_ht,
            'hits5': hits5_ht,
            'hits10': hits10_ht,
            'num': num_ht
        },
        'r': {
            'mrr': mrr_r,
            'hits1': hits1_r,
            'hits3': hits3_r,
            'hits5': hits5_r,
            'hits10': hits10_r,
            'num': num_r
        }
    }

    for arity in range(2, dataset_max_arity+1):
        eval_result['%d-r' % arity] = {
            'mrr': mrr_nr[arity],
            'hits1': hits1_nr[arity],
            'hits3': hits3_nr[arity],
            'hits5': hits5_nr[arity],
            'hits10': hits10_nr[arity],
            'num_nr': num_nr[arity]
        }
        eval_result['%d-ht' % arity] = {
            'mrr': mrr_nht[arity],
            'hits1': hits1_nht[arity],
            'hits3': hits3_nht[arity],
            'hits5': hits5_nht[arity],
            'hits10': hits10_nht[arity],
            'num_nht': num_nht[arity]
        }
        if arity == 2:
            continue
        else:
            eval_result['%d-a' % arity] = {
                'mrr': mrr_na[arity],
                'hits1': hits1_na[arity],
                'hits3': hits3_na[arity],
                'hits5': hits5_na[arity],
                'hits10': hits10_na[arity],
                'num_na': num_na[arity]
            }
            eval_result['%d-v' % arity] = {
                'mrr': mrr_nv[arity],
                'hits1': hits1_nv[arity],
                'hits3': hits3_nv[arity],
                'hits5': hits5_nv[arity],
                'hits10': hits10_nv[arity],
                'num_nv': num_nv[arity]
            }
    
    max_deg = len(_deg_ranks)-1
    for deg in range(0, max_deg+1):
        if len(_deg_ranks[deg])==0:
            continue 
        else:
            eval_result['entity degree = %d' % deg] = {
            'mrr': np.mean(1.0 / np.array(_deg_ranks[deg], dtype=np.float)),
            'num_ent': len(_deg_ranks[deg])
        }

    with open(eval_result_file, "w") as fw:
        fw.write(json.dumps(eval_result, indent=4) + "\n")

    return eval_result
