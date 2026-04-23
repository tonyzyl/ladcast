# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_geom_dataset
from configs.datasets_config import geom_with_h
import copy
import utils
import argparse
import wandb
from os.path import join
from qm9.models import get_optim, get_scheduler, get_model
from equivariant_diffusion import en_diffusion

from equivariant_diffusion import utils as diffusion_utils
import torch
import time
import pickle

from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save


parser = argparse.ArgumentParser(description='e3_diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')

parser.add_argument('--datadir', type=str, default=None,
                    help='Give the path to the directory containing the data')

# Training complexity is O(1) (unaffected), but sampling complexity O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--use_amsgrad', action="store_true")  # default from EDM
parser.add_argument('--weight_decay', type=float, default=1e-12)  # default from EDM

parser.add_argument('--scheduler', type=str, default=None)  # default from EDM
parser.add_argument('--num_warmup_steps', type=int, default=30000)  # default from EDM
parser.add_argument('--num_training_steps', type=int, default=350000)  # default from EDM

parser.add_argument('--clipping_type', type=str, default="queue")
parser.add_argument('--max_grad_norm', type=float, default=1.5)

parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')

# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=192,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args

parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='geom',
                    help='dataset name')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=50)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='multiple arguments can be passed, '
                         'including: homo | onehot | lumo | num_atoms | etc. '
                         'usage: "--conditioning H_thermo homo onehot H_thermo"')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0,           # TODO
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=20,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')

parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=5000)
parser.add_argument('--normalization_factor', type=float,
                    default=100, help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean" aggregation for the graph network')
parser.add_argument('--filter_molecule_size', type=int, default=None,
                    help="Only use molecules below this size.")
parser.add_argument('--sequential', action='store_true',
                    help='Organize data by size to reduce average memory usage.')     # This is for GNNs, keep it off

# -------- sym_diff args -------- #
parser.add_argument("--xh_hidden_size", type=int, default=128, help="config for DDG")
parser.add_argument("--K", type=int, default=128, help="config for DDG")

parser.add_argument("--enc_hidden_size", type=int, default=64, help="config for DDG")
parser.add_argument("--enc_depth", type=int, default=4, help="config for DDG")
parser.add_argument("--enc_num_heads", type=int, default=4, help="config for DDG")
parser.add_argument("--enc_mlp_ratio", type=float, default=4, help="config for DDG")

parser.add_argument("--dec_hidden_features", type=int, default=32, help="config for DDG")

parser.add_argument("--hidden_size", type=int, default=256, help="config for DDG")
parser.add_argument("--depth", type=int, default=6, help="config for DDG")
parser.add_argument("--num_heads", type=int, default=4, help="config for DDG")
parser.add_argument("--mlp_ratio", type=float, default=2.0, help="config for DDG")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="config for DDG")

parser.add_argument("--enc_concat_h", action="store_true", help="config for DDG")

parser.add_argument("--noise_dims", type=int, default=16, help="config for DDG")
parser.add_argument("--noise_std", type=float, default=1.0, help="config for DDG")

parser.add_argument("--mlp_type", type=str, default="swiglu", help="config for DDG")

parser.add_argument("--fix_qr", action="store_true", help="Whether to change the use of QR")


args = parser.parse_args()
data_file = args.datadir

if args.remove_h:
    raise NotImplementedError('Remove H not implemented.')
else:
    dataset_info = geom_with_h

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

split_data = build_geom_dataset.load_split_data(data_file, val_proportion=0.1, test_proportion=0.1, filter_size=args.filter_molecule_size)
transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, device, args.sequential)
dataloaders = {}
for key, data_list in zip(['train', 'val', 'test'], split_data):
    dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform)
    shuffle = (key == 'train') and not args.sequential

    # Sequential dataloading disabled for now.
    dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
        sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
        shuffle=shuffle)
del split_data

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

# If resume training, load the arguments from the previous run.
if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr

    # Load the arguments from the previous run
    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    args.resume = resume
    args.break_train_epoch = False
    args.exp_name = exp_name  # add _resume to the name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    print(args)

utils.create_folders(args)
print(args)

# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'name': args.exp_name, 'project': 'e3_diffusion_geom_drugs', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

data_dummy = next(iter(dataloaders['train']))


# For conditional generation - ignore
if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloader_train=dataloaders['train'])
model = model.to(device)
optim = get_optim(args, model)
scheduler = get_scheduler(args, optim)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # add large value that will be flushed.


def main():
    # If resuming, load the model and optimizer state dicts.
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'))
        # dequantizer_state_dict = torch.load(join(args.resume, 'dequantizer.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

        if args.scheduler is not None:
            scheduler_state_dict = torch.load(join(args.resume, 'scheduler.npy'))
            scheduler.load_state_dict(scheduler_state_dict)        

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1 and args.cuda:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = diffusion_utils.EMA(args.ema_decay)

        if args.resume is not None:
            ema_state_dict = torch.load(join(args.resume, 'generative_model_ema.npy'))
            model_ema.load_state_dict(ema_state_dict)        

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        wandb.log({"Epoch": epoch}, commit=True)
        print(f"--- Epoch {epoch} ---")        
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, scheduler=scheduler, prop_dist=prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0 and epoch != start_epoch: 
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                if args.com_free:
                    wandb.log(model.log_info(), commit=True)  # should be constant for l2                

            if not args.break_train_epoch:
                validity_dict = analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                                dataset_info=dataset_info, device=device,
                                                prop_dist=prop_dist, n_samples=args.n_stability_samples)
            nll_val = test(args=args, loader=dataloaders['val'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if scheduler is not None:
                        utils.save_model(scheduler, 'outputs/%s/scheduler.npy' % args.exp_name)                    
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
