import argparse
import os
from pathlib import Path
import random
import re

import gymnasium as gym
import gymnasium_robotics
from mpi4py import MPI
import numpy as np
import torch
import wandb

from rl_modules.ddpg_agent import DDPG
from rl_modules.gcsl_agent import GCSL
from rl_modules.gofar_agent import GoFAR

def infer_imbalance_from_path(buffer_path):
    """Infer the intervention dimension, target subtype, and ratio from a path."""
    if buffer_path is None:
        return "baseline", "all", "1"

    p = Path(buffer_path)
    parent = p.parent.name.lower()
    if parent in ["bias", "coverage", "reachability"]:
        imb_type = parent
    else:
        imb_type = "baseline"

    stem = p.stem
    target = "all"
    ratio = "1"
    m_ratio = re.search(r"_r([0-9.]+)", stem)
    if m_ratio:
        ratio = m_ratio.group(1)

    parts = stem.split("_")
    for i, w in enumerate(parts):
        if w.startswith("r") and i > 0:
            target = parts[i-1]
            break
    if target not in ["left","right","inner","outer","short","long","all"]:
        if imb_type in parts:
            j = parts.index(imb_type)
            if j+1 < len(parts):
                target = parts[j+1]

    return imb_type, target, ratio


def build_run_name(args):
    imb_tag = f"{args.imb_type}-{args.imb_target}-r{args.imb_ratio}"
    data_tag = getattr(args, "data_tag", "single")
    relabel_tag = f"relabel{args.relabel_percent}"
    reward_tag = args.reward_type
    if args.reward_type == "disc":
        reward_tag = f"{args.disc_iter}disc{args.disc_lambda}"
    elif args.reward_type == "binary":
        reward_tag = f"{args.reward_type}{args.threshold}"

    run_name = (
        f"{args.env_id}-"
        f"{imb_tag}-"
        f"{data_tag}-"
        f"{args.expert_percent}-{args.random_percent}-"
        f"{args.method}-"
        f"{reward_tag}-{relabel_tag}-"
        f"{args.seed}"
    )
    return run_name

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument(
        '--env',
        type=str,
        default='FetchPush',
        choices=['FetchPush', 'PointMaze'],
        help='evaluation environment',
    )
    parser.add_argument('--n-epochs', type=int, default=800, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=10, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=20, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(2e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', default=True, type=boolean, help='if use gpu do the acceleration')
    parser.add_argument('--device', default=0, type=int, help='gpu device number')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=1, help='the rollouts per mpi')

    # hyperparameters that need to be changed
    parser.add_argument('--eval', default=True, type=boolean)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--method',
        default='gofar',
        choices=['ddpg', 'gcsl', 'wgcsl', 'gofar'],
    )
    parser.add_argument('--f', default='chi', type=str)
    parser.add_argument('--online', default=False, type=boolean)

    parser.add_argument('--relabel', default=False, type=boolean)
    parser.add_argument('--relabel_percent', default=0.0, type=float)

    parser.add_argument('--reward_type', default='binary', type=str)
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--disc_iter', type=int, default=20)
    parser.add_argument('--disc_lambda', type=float, default=0.01)

    parser.add_argument('--expert_percent', type=float, default=0.5, help='the expert coefficient')
    parser.add_argument('--random_percent', type=float, default=0.5, help='the random coefficient')
    
        # ===== NEW: dynamic buffer loading =====
    parser.add_argument("--buffer-path", type=str, default=None,
                        help="single buffer pkl for training (no mixture)")
    parser.add_argument("--buffer-expert", type=str, default=None,
                        help="expert buffer pkl for mixture training")
    parser.add_argument("--buffer-random", type=str, default=None,
                        help="random buffer pkl for mixture training")
    parser.add_argument("--use-mixture", action="store_true",
                        help="if set, load expert+random mixture instead of single buffer")

    # optional: if you want to override auto-inferred imbalance info
    parser.add_argument("--imbalance-type", type=str, default=None,
                    choices=["baseline", "bias", "coverage", "reachability"])
    parser.add_argument("--imbalance-target", type=str, default=None,
                        help="left/right/inner/outer/short/long")
    parser.add_argument("--imbalance-ratio", type=str, default=None,
                        help="e.g. 0.1, 2, 5 (meaning r:1)")

    parser.add_argument(
        "--wandb-project",
        default="trajectory-type-imbalance-offline-gcrl",
        help="W&B project name",
    )
    parser.add_argument("--wandb-entity", default=None, help="optional W&B entity")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-group", default=None)

    

    args = parser.parse_args()

    return args

def get_env_params(env):
    obs, _ = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'action_space': env.action_space
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def get_full_envname(name):
    dic = {
        'FetchPush': 'FetchPush-v1',
        'PointMaze': 'PointMaze_Medium-v3',
    }
    if name in dic.keys():
        return dic[name]
    else:
        return name

def get_method_params(args):
    
        # ===== NEW: sanity check for buffer args =====
    if args.use_mixture:
        if args.buffer_expert is None or args.buffer_random is None:
            raise ValueError("use --use-mixture with --buffer-expert and --buffer-random")
        infer_path = args.buffer_expert
        data_tag = "mixture"
    else:
        if args.buffer_path is None:
            print("[WARN] --buffer-path not provided, treat as baseline/online run.")
        infer_path = args.buffer_path
        data_tag = "single"

    # ===== NEW: infer imbalance labels from path (unless user overrides) =====
    imb_type, imb_target, imb_ratio = infer_imbalance_from_path(infer_path)

    if args.imbalance_type is not None:
        imb_type = args.imbalance_type
    if args.imbalance_target is not None:
        imb_target = args.imbalance_target
    if args.imbalance_ratio is not None:
        imb_ratio = args.imbalance_ratio

    args.imb_type = imb_type
    args.imb_target = imb_target
    args.imb_ratio = imb_ratio
    args.data_tag = data_tag  # single or mixture

    
    
    if args.online:
        args.n_batches = 40
        args.n_cycles = 50
        
    if args.method == 'ddpg':
        args.lr_actor = 0.001
        args.lr_critic = 0.001
    elif 'gcsl' in args.method:
        args.lr_actor = 5e-4
        args.lr_critic = 5e-4

    if 'gcsl' in args.method:
        args.relabel_percent = 1.0
    if 'gofar' in args.method:
        args.relabel = False 
        args.reward_type = 'disc' 

def launch(args):
    get_method_params(args)
    args.use_disc = True if args.reward_type =='disc' else False

    args.env_id = get_full_envname(args.env)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env_id)

    if args.relabel == False:
        args.relabel_percent = 0.
        
        
    args.run_name = build_run_name(args)
    run_name = args.run_name


    if MPI.COMM_WORLD.Get_rank() == 0:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=run_name,
            group=args.wandb_group or f"{args.env}/{args.imb_type}",
            job_type=args.method,
            tags=[
                args.env_id,
                args.imb_type,
                args.imb_target,
                f"r{args.imb_ratio}",
                args.method,
                args.data_tag,
            ],
            config=vars(args),
        )



    # set random seeds for reproduce
    env.reset(seed=args.seed + MPI.COMM_WORLD.Get_rank()) # env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    
    # get the environment parameters
    
    # ===== NEW: limit torch CPU threads =====
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    env_params = get_env_params(env)

    # create agent
    if args.method == 'ddpg':
        trainer = DDPG(args, env, env_params)
    elif args.method == 'gofar':
        trainer = GoFAR(args, env, env_params)
    elif 'gcsl' in args.method:
        trainer = GCSL(args, env, env_params)
    else:
        raise NotImplementedError
    print(run_name)
    
    # do offline goal-conditioned rl 
    trainer.learn(evaluate_agent=args.eval)

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    # get the params
    launch(args)
