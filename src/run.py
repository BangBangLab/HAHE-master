from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import torch.utils.data.dataloader as DataLoader


from reader.vocab_reader import Vocabulary
from reader.data_reader import  MultiDataset,read_examples,prepare_adj_info
from model.hahe_model import HAHEModel
from new_evaluation import generate_ground_truth, new_batch_evaluation, new_compute_metrics
from utils.args import ArgumentGroup, print_arguments
from utils.process_dataset import process_dataset


torch.set_printoptions(precision=8)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

# yapf: disable
parser = argparse.ArgumentParser()
model_g = ArgumentGroup(parser, "model", "model and checkpoint configuration.")
model_g.add_arg("num_hidden_layers",       int,    12,        "Number of hidden layers.")
model_g.add_arg("num_attention_heads",     int,    8,         "Number of attention heads.")
model_g.add_arg("hidden_size",             int,    256,       "Hidden size.")
model_g.add_arg("intermediate_size",       int,    512,       "Intermediate size.")
model_g.add_arg("hidden_dropout_prob",     float,  0.2,       "Hidden dropout ratio.")
model_g.add_arg("attention_dropout_prob",  float,  0.2,       "Attention dropout ratio.")
model_g.add_arg("initializer_range",       float,  0.02,      "Initializer range.")
model_g.add_arg("num_edges",               int,    14,
                "Number of edge types, typically fixed to 5: no edge (0), relation-subject (1),"
                "relation-object (2), relation-attribute (3), attribute-value (4).")
model_g.add_arg("entity_soft_label",       float,  0.8,       "Label smoothing rate for masked entities.")
model_g.add_arg("relation_soft_label",     float,  0.9,       "Label smoothing rate for masked relations.")
model_g.add_arg("HGNN_layer_num",     int,  2,       "Number of HGNN layers.")
model_g.add_arg("checkpoint_dir",             str,    "./new_ckpts",   "Path to save checkpoints.")
model_g.add_arg("eval_dir",             str,    "./src/eval_result_new",   "Path to save eval_result.")
model_g.add_arg("encoder_order",            str,    "true_GL",       "encoder_order") 
model_g.add_arg("L_config",                 str,        "L_node_edge",       "L encoder config")
train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("batch_size",        int,    64,                   "Batch size.")
train_g.add_arg("epoch",             int,    300,                    "Number of training epochs.")
train_g.add_arg("learning_rate",     float,  5e-4,                   "Learning rate with warmup.")

train_g.add_arg("warmup_proportion", float,  0.1,                    "Proportion of training steps for lr warmup.")
train_g.add_arg("weight_decay",      float,  0.01,                   "Weight decay rate for L2 regularizer.")
train_g.add_arg("save_steps",          int,   20,                  "save_steps")
train_g.add_arg("num_workers",      int,    2,                     "num_workers")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("dataset",             str,   "WD50K",   "Dataset")#JF17K,Wikipeople,WD50K
log_g.add_arg("print_steps",         int,   10,      "print steps")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("gpu_index",             int,            2,      "gpu index")
run_type_g.add_arg("do_train",                     bool,   True, "Whether to perform training.")
run_type_g.add_arg("do_predict",                   bool,   True, "Whether to perform prediction.")
args = parser.parse_args()
# yapf: enable.

def main(args):
    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")
    config = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)

    if args.use_cuda:
        device = torch.device("cuda")
        config["device"]="cuda"
    else:
        device = torch.device("cpu")
        config["device"]="cpu"

    config = process_dataset(config['dataset'], config) 

    vocabulary = Vocabulary(
        vocab_file=args.vocab_path,
        num_relations=args.num_relations,
        num_entities=args.vocab_size - args.num_relations - 2)

    
    examples,total_instance = read_examples(args.train_file)
    train_data_reader=MultiDataset(
        vocabulary,
        examples,
        args.max_arity,
        args.max_seq_len,
        False)

    num_train_instances = total_instance
    max_train_steps = args.epoch * ((num_train_instances // args.batch_size )+1)
    warmup_steps = int(max_train_steps * args.warmup_proportion)
    logger.info("Num train instances: %d" % num_train_instances)
    logger.info("Max train steps: %d" % max_train_steps)
    logger.info("Num warmup steps: %d" % warmup_steps)
    steps_per_epoch=num_train_instances//args.batch_size+1
    logger.info("steps_per_epoch: %d" % steps_per_epoch)
    config['hyperedge']=len(examples)

    train_pyreader = DataLoader.DataLoader(
        train_data_reader, 
        batch_size= args.batch_size, 
        shuffle = True, 
        drop_last=False, 
        num_workers=args.num_workers)

    adj_info= prepare_adj_info(train_data_reader, device)


    test_examples,test_total_instance = read_examples(args.predict_file) 
    test_data_reader=MultiDataset(
        vocabulary,
        test_examples,
        args.max_arity,
        args.max_seq_len,
        False)

    num_test_instances = test_total_instance
    logger.info("Num test instances: %d" % num_test_instances)
    test_pyreader = DataLoader.DataLoader(
        test_data_reader, 
        batch_size= args.batch_size, 
        shuffle = False, 
        drop_last=False,
        num_workers=args.num_workers)


    if args.do_train:

        train_writer_dir = "log/train"

        hahe_model = HAHEModel(adj_info,config=config).to(device)
        optimizer=torch.optim.Adam(hahe_model.parameters(),lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps:steps/warmup_steps if steps<=warmup_steps else (max_train_steps-steps)/(max_train_steps-warmup_steps))

        steps = 0
        best_step=0
        best_loss=10000.0
        total_cost=0.0
        average_loss=0.0
        training_range = tqdm(range(args.epoch))
        for epoch in training_range:          
            for i, item in enumerate(train_pyreader):
                steps+=1 
                input_ids,\
                    input_mask,\
                    mask_position,\
                    mask_label,\
                    mask_type,\
                    edge_labels=item

                input_ids=input_ids.to(device).long()
                input_mask=input_mask.to(device).float()
                mask_position=mask_position.to(device).long()
                mask_label=mask_label.to(device).squeeze().long()
                mask_type=mask_type.to(device).long()
                edge_labels=edge_labels[0].to(device).long() 
               
                hahe_model.train() 
                scheduled_lr=optimizer.state_dict()['param_groups'][0]['lr']
                optimizer.zero_grad()
                loss,fc_out = hahe_model(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    edge_labels=edge_labels,
                    mask_pos=mask_position,
                    mask_label=mask_label, 
                    mask_type=mask_type)               
                loss.backward()
                optimizer.step()
                scheduler.step()
                if args.weight_decay >= 0:
                    for param in hahe_model.parameters():
                        if param.requires_grad:
                            param_copy = param.data.detach()
                            param.data = param_copy - param_copy * args.weight_decay * scheduled_lr

                hahe_model.eval()
                with torch.no_grad():
                    total_cost+=loss
                    average_loss=total_cost/steps                    
                    if loss < best_loss and steps>10:
                        best_loss=loss
                        best_step=steps
                        torch.save(hahe_model.state_dict(),os.path.join(args.checkpoint_dir , "ckpts"+args.dataset +"-best-DIM"+str(args.hidden_size)+ ".ckpt"))                   
                    training_range.set_description("Epoch %d | Steps %d | lr: %f | loss: %f | best_loss: %f at step%d | average_loss: %f "  % (epoch,steps, scheduled_lr,loss,best_loss,best_step,average_loss))
            if epoch%args.save_steps==0 and epoch!=0:
                torch.save(hahe_model.state_dict(),os.path.join(args.checkpoint_dir , "ckpts"+args.dataset +"-best-DIM"+str(args.hidden_size)+ ".ckpt"))
            if epoch%args.print_steps == 0 and epoch != 0:
                print_predict(hahe_model, test_pyreader, test_data_reader, vocabulary, device, train_writer_dir) 

        torch.save(hahe_model.state_dict(),os.path.join(args.checkpoint_dir , "ckpts"+args.dataset +"-last-DIM"+str(args.hidden_size)+ ".ckpt"))            




    if args.do_predict:

        degree_table = np.zeros(shape=(args.vocab_size), dtype=np.int64)

        adj2_start = adj_info[1]['indices'][0, :].cpu().numpy()

        for id in range(args.vocab_size):
            degree_table[id] = np.count_nonzero(np.isin(adj2_start, [id]))

        max_deg = np.amax(degree_table)



        test_writer_dir = "log/test"
    


        if args.do_train:
            hahe_model = HAHEModel(adj_info,config=config).to(device)
            hahe_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir , "ckpts"+args.dataset +"-last-DIM"+str(args.hidden_size)+ ".ckpt"))) 
        else:
            hahe_model = HAHEModel(adj_info,config=config).to(device)
            hahe_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir , "new_ckpts"+args.dataset +"-best-DIM"+str(args.hidden_size)+ ".ckpt"))) 

        num_params = sum(param.numel() for param in hahe_model.parameters())
        print(num_params)     

        max_perdict_steps=num_test_instances//args.batch_size
        logger.info("max_perdict_steps: %d" % max_perdict_steps)


        print_predict(hahe_model, test_pyreader, test_data_reader, vocabulary, device, test_writer_dir, degree_table)




def predict(model, test_pyreader,  all_features, vocabulary, device, degree_table):
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)
    eval_result_file = os.path.join(args.eval_dir, "eval_result.json")

    gt_dict = generate_ground_truth(
        ground_truth_path=args.ground_truth_path,
        vocabulary=vocabulary,
        max_arity=args.max_arity,
        max_seq_length=args.max_seq_len)

    step = 0
    global_idx = 0

    max_deg = np.amax(degree_table)
    degree_ranks = [[] for i in range(0, max_deg+1)]

    ent_lst = []
    rel_lst = []
    _r_lst = [[] for i in range(args.max_arity+1)]
    _ht_lst = [[] for i in range(args.max_arity+1)]
    _a_lst = [[] for i in range(args.max_arity+1)]
    _v_lst = [[] for i in range(args.max_arity+1)]

    model.eval()
    with torch.no_grad():
        #while steps < max_train_steps:
        predict_range=tqdm(enumerate(test_pyreader))
        for i, item in predict_range:
            input_ids,\
                input_mask,\
                mask_position,\
                mask_label,\
                mask_type,\
                edge_labels=item

            input_ids=input_ids.to(device).long()
            input_mask=input_mask.to(device).float()
            mask_position=mask_position.to(device).long()
            mask_label=mask_label.to(device).squeeze().long()
            mask_type=mask_type.to(device).int()
            edge_labels=edge_labels[0].to(device).long() 

            batch_results = []
            _,np_fc_out = model(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        edge_labels=edge_labels,
                        mask_pos=mask_position, 
                        mask_label=mask_label, 
                        mask_type=mask_type)
            batch_results = np_fc_out.cpu().numpy()



            ent_ranks, rel_ranks, _r_ranks, _ht_ranks, _a_ranks, _v_ranks, _deg_ranks = new_batch_evaluation(
                global_idx, batch_results, all_features, gt_dict, args.max_arity, degree_table)



            ent_lst.extend(ent_ranks)
            rel_lst.extend(rel_ranks)
            for arity in range(2, args.max_arity+1):
                _r_lst[arity].extend(_r_ranks[arity])
                _ht_lst[arity].extend(_ht_ranks[arity])
                _a_lst[arity].extend(_a_ranks[arity])
                _v_lst[arity].extend(_v_ranks[arity])
            
            for deg in range(0, max_deg+1):
                degree_ranks[deg].extend(_deg_ranks[deg])


            predict_range.set_description("Processing prediction steps: %d | examples: %d" % (step, global_idx))
            step += 1
            global_idx += np_fc_out.size(0)

    eval_result = new_compute_metrics(
        ent_lst=ent_lst,
        rel_lst=rel_lst,
        _r_lst=_r_lst,
        _ht_lst=_ht_lst,
        _a_lst=_a_lst,
        _v_lst=_v_lst,
        _deg_ranks = degree_ranks,
        eval_result_file=eval_result_file,
        dataset_max_arity=args.max_arity
    )

    return eval_result

def print_predict(hahe_model, test_pyreader, test_data_reader, vocabulary, device, writer_dir, degree_table):


    vocab_emb = hahe_model.node_embedding.weight.data.cpu().numpy()
    np.save(os.path.join(writer_dir, "vocab_emb"), vocab_emb) 

    hyperedge_emb = hahe_model.hyperedge_embedding.weight.data.cpu().numpy()
    np.save(os.path.join(writer_dir, "hyperedge_emb"), hyperedge_emb)




    eval_performance = predict(
        model=hahe_model,
        test_pyreader=test_pyreader,
        all_features=test_data_reader.features,
        vocabulary=vocabulary,
        device=device,
        degree_table=degree_table)


    all_entity = "ENTITY\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['entity']['mrr'],
        eval_performance['entity']['hits1'],
        eval_performance['entity']['hits3'],
        eval_performance['entity']['hits5'],
        eval_performance['entity']['hits10'])

    all_relation = "RELATION\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['relation']['mrr'],
        eval_performance['relation']['hits1'],
        eval_performance['relation']['hits3'],
        eval_performance['relation']['hits5'],
        eval_performance['relation']['hits10'])

    all_ht = "HEAD/TAIL\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['ht']['mrr'],
        eval_performance['ht']['hits1'],
        eval_performance['ht']['hits3'],
        eval_performance['ht']['hits5'],
        eval_performance['ht']['hits10'])

    all_r = "PRIMARY_R\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['r']['mrr'],
        eval_performance['r']['hits1'],
        eval_performance['r']['hits3'],
        eval_performance['r']['hits5'],
        eval_performance['r']['hits10'])

    logger.info("\n-------- Evaluation Performance --------\n%s\n%s\n%s\n%s\n%s" % (
        "\t".join(["TASK", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
        all_ht, all_r, all_entity, all_relation))
    


if __name__ == '__main__':
    print_arguments(args)
    main(args)