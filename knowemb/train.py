import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE,TransE,DistMult
from openke.module.loss import MarginLoss, SoftplusLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse
import wandb

parser = argparse.ArgumentParser()
#parser.add_argument('run_name', metavar='R', type=str)
#parser.add_argument('project_name', metavar='P', type=str)
parser.add_argument('--model', default='RotatE', type=str)
parser.add_argument('--data_path', default='datas/Prescriptions/',type=str)
parser.add_argument('--lr',default=1e-2, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--bs', default=256, type=int)
parser.add_argument('--ns', default=25, type=int)
parser.add_argument('--dim', default=128, type=int)
parser.add_argument('--p', default=1, type=int)
parser.add_argument('--margin', default=5.0, type=float)
args = parser.parse_args()

# WandB setup
wandb.init(config=vars(args),project=args.model)
wandb.run.name = wandb.run.id
wandb.run.save()
# wandb.config.model = args.model
# wandb.config.lr = args.lr
# wandb.config.epochs = args.epochs
# wandb.config.nBatchs = args.bs
# wandb.config.nNegs = args.ns
# wandb.config.hid_dim = args.dim
# wandb.config.p_norm = args.p
# wandb.config.margin= args.margin

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = args.data_path,
	batch_size = args.bs,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = args.ns,
	neg_rel = 0)

# dataloader for test
# test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
if args.model == 'TransE':
	model_init = TransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = args.dim,
		p_norm = args.p,
		norm_flag = True)
	# define the loss function
	model = NegativeSampling(
		model=model_init,
		loss=MarginLoss(margin=args.margin),
		batch_size=train_dataloader.get_batch_size()
	)
elif args.model == 'DistMult':
	model_init = DistMult(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = args.dim)
	model = NegativeSampling(
		model=model_init,
		loss=SoftplusLoss(),
		batch_size=train_dataloader.get_batch_size(),
		regul_rate=1.0
	)
elif args.model == 'RotatE':
	model_init = RotatE(
		ent_tot=train_dataloader.get_ent_tot(),
		rel_tot=train_dataloader.get_rel_tot(),
		dim=args.dim,
		margin=args.margin
	)
	model = NegativeSampling(
		model=model_init,
		loss=SigmoidLoss(),
		batch_size=train_dataloader.get_batch_size(),
		regul_rate=1.0
	)
else:
	raise NotImplementedError

#print(wandb.run.name)
# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = args.epochs, alpha = args.lr, use_gpu = True, save_steps=int(args.bs/10),checkpoint_dir='trained_model/{}'.format(wandb.run.name),wandb=wandb)
trainer.run()
#transe.save_checkpoint('./trained_models/transe.ckpt')

# test the model
# transe.load_checkpoint('./checkpoint/transe.ckpt')
# tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction(type_constrain = False)