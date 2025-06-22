import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from collections import OrderedDict
from copy import deepcopy
import argparse
import logging
import os
from train_util import TrainLoop
from glob import glob
from diffusers.models import AutoencoderKL
from time import time
import wandb_utils
import logging
from dataset import ADTDataset
from models import SiT_models, WeightFunc
from model_zigma import ZigMa
   
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger



@th.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag





def create_zigma(image_size, use_fp16=False):
    return ZigMa(
        in_channels=3,
        embed_dim=256,
        depth=12,
        img_dim=image_size,
        patch_size=2,
        has_text=False,
        num_classes=-1,
        drop_path_rate=0.1,
        n_context_token=0,
        d_context=0,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=use_fp16,
        initializer_cfg=None,
        scan_type="zigzagN8",
        video_frames=0,
        tpe=False,  # apply temporal positional encoding for video-related task
        device="cuda",
        use_pe=0,
        use_jit=True,
        m_init=True,
        use_checkpoint=False,
        use_fp16=use_fp16,
        
    )

    
    
    
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def main(args):
    
    
    #setting up DDP
    assert th.cuda.is_available()
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % th.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    th.manual_seed(seed)
    th.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())
    
    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = "test"  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        
        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)
    
    #setting up the models for CD
    #state_dict = find_model(args.ckpt_path)
    #model = SiT_models[args.model](
        #input_size=args.image_size,
        #num_classes=args.num_classes
    #)
    model = create_zigma(args.image_size)
    pretrain_model = deepcopy(model).to(device)
    avg_model = deepcopy(model).to(device)
    
    weight_fn = WeightFunc(128)
    
    #pretrain_model.load_state_dict(state_dict)
    model = DDP(model.to(device), device_ids=[rank],find_unused_parameters=True)
    weight_fn = DDP(weight_fn.to(device), device_ids=[rank],find_unused_parameters=True)
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    opt = th.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    dataset = ADTDataset(args.data_path, args.image_size)
    train_fn = TrainLoop(
                sigma_data=0.5,
                P_mean=-0.8,
                P_std=1.6,
                pretrain_model=pretrain_model,
                batch_size=local_batch_size,
                c=0.1
    )
    
    
    loader = DataLoader(dataset=dataset,
                        batch_size=local_batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True
    )
    requires_grad(avg_model, False)
    update_ema(avg_model, model.module, decay=0)
    weight_fn.train()
    model.train()
    avg_model.eval()
    pretrain_model.eval()
    iters = 0
    running_loss = 0
    log_steps = 0
    train_steps = 0
    logger.info("Starting training")
    for i in loader:
        
        x0 = []
        x1 = i
        x = x1[1]
        x = x.to(device)
        #with th.no_grad():
            # Map input images to latent space + normalize latents:
            #x1 = vae.encode(x1).latent_dist.sample().mul_(0.18215)
            #norm = torchvision.transforms.Normalize((1.56, -0.695, 0.483, 0.729),(0.5/5.27, 0.5/5.91, 0.5/4.21, 0.5/4.31))
            #x1 = norm(x1)
        r = min(1., iters/args.H)
        with th.autograd.set_detect_anomaly(True):
            loss = train_fn.train_loss(x1=x, r=r, model=model, avg_model=avg_model, weight_fn=weight_fn).mean()
        
            loss.backward()
            opt.step()
            update_ema(avg_model, model.module)
            opt.zero_grad()
        running_loss += loss.item()
        log_steps += 1
        train_steps += 1
        if train_steps % args.log_every == 0:
            # Measure training speed:
            th.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = th.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
            if args.wandb:
                wandb_utils.log(
                    { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                    step=train_steps
                )
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()
            
        print("completed iteration")
            
            
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-S/8")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--H", type=int, default=10000)
    
    
    args = parser.parse_args()
    main(args)