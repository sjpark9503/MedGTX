from absl import app
from absl import flags
import data
import metric
import model as model_definition
import os
import storage
import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard
from typing import Tuple
from tqdm import tqdm
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", default='TransE', help="Learning rate value.")
flags.DEFINE_float("lr", default=0.01, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=128, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64, help="Maximum batch size during model validation.")
flags.DEFINE_integer("vector_length", default=50, help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0, help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer("norm", default=2, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=2000, help="Number of training epochs.")
flags.DEFINE_string("dataset_path", default="./synth_data", help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")
flags.DEFINE_integer("validation_freq", default=10, help="Validate model every X epochs.")
flags.DEFINE_string("checkpoint_path", default="", help="Path to model checkpoint (by default train from scratch).")
flags.DEFINE_string("tensorboard_log_dir", default="./runs", help="Path for tensorboard log directory.")
flags.DEFINE_bool("do_test", default=False, help="Path for tensorboard log directory.")

HITS_AT_1_SCORE = float
HITS_AT_3_SCORE = float
HITS_AT_10_SCORE = float
MRR_SCORE = float
METRICS = Tuple[HITS_AT_1_SCORE, HITS_AT_3_SCORE, HITS_AT_10_SCORE, MRR_SCORE]


def test(model: torch.nn.Module, data_generator: torch_data.DataLoader, entities_count: int,
         device: torch.device,
         ) -> METRICS:
    examples_count = 0.0
    hits_at_1 = 0.0
    hits_at_3 = 0.0
    hits_at_10 = 0.0
    mrr = 0.0

    entity_ids = torch.arange(end=entities_count, device=device).unsqueeze(0)
    for head, relation, tail in data_generator:
        current_batch_size = head.size()[0]

        head, relation, tail = head.to(device), relation.to(device), tail.to(device)
        all_entities = entity_ids.repeat(current_batch_size, 1)
        heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails = tail.reshape(-1, 1).repeat(1, all_entities.size()[1])

        # Check all possible tails
        triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        tails_predictions = model.predict(triplets).reshape(current_batch_size, -1)
        # Check all possible heads
        triplets = torch.stack((all_entities, relations, tails), dim=2).reshape(-1, 3)
        heads_predictions = model.predict(triplets).reshape(current_batch_size, -1)

        # Concat predictions
        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((tail.reshape(-1, 1), head.reshape(-1, 1)))

        hits_at_1 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=1)
        hits_at_3 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=3)
        hits_at_10 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=10)
        mrr += metric.mrr(predictions, ground_truth_entity_id)

        examples_count += predictions.size()[0]

    hits_at_1_score = hits_at_1 / examples_count * 100
    hits_at_3_score = hits_at_3 / examples_count * 100
    hits_at_10_score = hits_at_10 / examples_count * 100
    mrr_score = mrr / examples_count * 100
    wandb.log({'Hits@1':hits_at_1_score,'Hits@3':hits_at_3_score,'Hits@10':hits_at_10_score,'MRR':mrr_score})

    return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score

def train (model, optimizer, device, epochs, train_generator, voca):
    start_epoch_id = 1
    step = 0
    best_score = 0.0
    if FLAGS.checkpoint_path:
        start_epoch_id, step, best_score = wandb.restore(FLAGS.checkpoint_path)
    model.train()

    # Training loop
    for epoch_id in range(start_epoch_id, epochs + 1):
        print("Starting epoch: ", epoch_id)
        loss_impacting_samples_count = 0
        samples_count = 0

        for positive_triples in tqdm(train_generator):
            positive_triples = positive_triples.to(device)

            # Preparing negatives.
            # Generate binary tensor to replace either head or tail. 1 means replace head, 0 means replace tail.
            head_or_tail = torch.randint(high=2, size=positive_triples[:,0].size(), device=device)
            random_entities = torch.randint(high=len(voca['node']), size=positive_triples[:,0].size(), device=device)
            broken_heads = torch.where(head_or_tail == 1, random_entities, positive_triples[:,0])
            broken_tails = torch.where(head_or_tail == 0, random_entities, positive_triples[:,2])
            negative_triples = torch.stack((broken_heads, positive_triples[:,1], broken_tails), dim=1)

            optimizer.zero_grad()

            loss, pd, nd = model(positive_triples, negative_triples)
            loss.mean().backward()

            wandb.log({'train_loss': loss.mean().item(),'pos_d':pd.sum().item(), 'neg_d':nd.sum().item()})

            loss = loss.data.cpu()
            loss_impacting_samples_count += loss.nonzero().size()[0]
            samples_count += loss.size()[0]

            optimizer.step()
            step += 1

        if epoch_id % FLAGS.validation_freq == 0:
            if FLAGS.do_test:
                model.eval()
                _, _, hits_at_10, _ = test(model=model, data_generator=validation_generator,
                                           entities_count=len(voca['node']),
                                           device=device,
                                           epoch_id=epoch_id, metric_suffix="val")
                score = hits_at_10
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_epochID = epoch_id
            else:
                best_epochID = epoch_id
        wandb.log({'num_impacting_samples':loss_impacting_samples_count})
    torch.save('trained_model/ckpt{}.pt', best_epochID)
    wandb.save('trained_model/*.pt')

    return best_model

def main(_):
    ## Only for stand-alone version

    wandb.init(project=FLAGS.project_name)
    wandb.config.lr = FLAGS.lr
    wandb.config.seed = FLAGS.seed
    wandb.config.batch_size = FLAGS.batch_size
    wandb.config.hidden_size = FLAGS.vector_length
    wandb.config.margin = FLAGS.margin
    wandb.config.p = FLAGS.norm
    wandb.config.epoch = FLAGS.epochs
    torch.random.manual_seed(FLAGS.seed)

    path = FLAGS.dataset_path

    batch_size = FLAGS.batch_size
    vector_length = FLAGS.vector_length
    margin = FLAGS.margin
    norm = FLAGS.norm
    learning_rate = FLAGS.lr

    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')
    train_generator, validation_generator, test_generator, voca = data.build_dataset(path, train_bs = batch_size, eval_bs = FLAGS.validation_batch_size)

    model = model_definition.TransE(entity_count=len(voca['node']), relation_count=len(voca['edge']), dim=vector_length,
                                    margin=margin,
                                    device=device, norm=norm)  # type: torch.nn.Module
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model = train(model, optimizer, device, FLAGS.epochs,train_generator, voca)
    print(model)

    # Testing the best checkpoint on test dataset
    if FLAGS.do_test:
        best_model = model.to(device)
        best_model.eval()
        scores = test(model=best_model, data_generator=test_generator, entities_count=len(voca['node']), device=device, epoch_id=1, metric_suffix="test")
        print("Test scores: ", scores)

if __name__ == '__main__':
    app.run(main)
