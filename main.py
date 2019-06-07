import multiprocessing
import copy
import glob
import os
import time
from collections import deque

import subprocess
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from contextlib import redirect_stdout

from tensorboardX import SummaryWriter
from contextlib import contextmanager
import signal
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot,td_plot
from bulletrobotgym.utils.blogging import log, suppress_logging,set_log_file
from bulletrobotgym.utils.comm import makedir_if_not_exists,suppress_stdout
from gym.envs.registration import register
register(
    id='tcn-push-v0',
    entry_point='bulletrobotgym.env_tcn:TcnPush',
)

args = get_args()
args.log_dir=os.path.expanduser(args.log_dir)

os.environ["OPENAI_LOGDIR"]=args.log_dir
os.environ["TCN_ENV_VID_LOG_FOLDER"]='train_vid'

os.environ['TCN_ENV_VID_LOG_INTERVAL'] = '100'
set_log_file(os.path.join(args.log_dir, "env.log"))


log.info(args)
assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    log.info("cuda_deterministic ")
else:
    torch.backends.cudnn.benchmark = True
try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

def _tb_task(path_tb,port):
    # with suppress_stdout():
        # with _start_subprocess('tensorboard --port={} --logdir={}'.format(port, path_tb)):
            # while True:
                # time.sleep(10)
    import tensorboard
    from tensorboard import default
    from tensorboard import program
    import logging

    class TensorBoardTool:
        '''Tensorboard V1.12 start'''
        def __init__(self, dir_path,port):
            self.dir_path = dir_path
            self.port=port
        def run(self):
            # Remove http messages
            log = logging.getLogger('werkzeug').setLevel(logging.ERROR)

            logging.getLogger("tensorflow").setLevel(logging.WARNING)
            # Start tensorboard server
            tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
            tb.configure(argv=[None, '--logdir', self.dir_path,'--port',str(self.port)])
            url = tb.launch()
            print('TensorBoard at %s \n' % url)
    # Tensorboard tool launch
    with suppress_stdout():
        tb_tool = TensorBoardTool(path_tb,port)
        tb_tool.run()

@contextmanager
def _start_subprocess(cmd):
    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    pro = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                           shell=True,
                           preexec_fn=os.setsid)
    try:
        yield pro
    except:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)  # Send the signal to all the process groups


def main():
    tb_path=os.path.join(
        os.path.expanduser(args.log_dir), "tensorboard_log")
    makedir_if_not_exists(tb_path)
    writer = SummaryWriter(tb_path)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # p = multiprocessing.Process(target=_tb_task,args=(tb_path,5013) ,daemon=True)
    # p.start()
    _tb_task(tb_path,port=5013)
    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    start = time.time()

    basline_rw_episode_rec=[]
    basline_rw_episode_mse=[]

    basline_rw_episode_tcn=[]
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'basline_rw_mse' in info:
                    basline_rw_episode_mse.append(info['basline_rw_mse'])
                    basline_rw_episode_rec.append(info['basline_rw_rec'])
                if 'basline_rw_tcn' in info:
                    basline_rw_episode_tcn.append(info['basline_rw_tcn'])

                if 'episode' in info.keys():
                    # episode is done
                    # add addisiotnal baseline rw
                    episode_rewards.append(info['episode']['r'])
                    writer.add_scalar('basline/rw_mse', np.sum(basline_rw_episode_mse), j)
                    writer.add_scalar('basline/rw_rec', np.sum(basline_rw_episode_rec), j)
                    if 'basline_rw_tcn' in info:
                        writer.add_scalar('basline/rw_tcn', np.sum(basline_rw_episode_tcn), j)
                    writer.add_scalar('basline/rw_push_dist', min(0.,info['basline_rw_push_dist']), j)
                    basline_rw_episode_mse=[]
                    basline_rw_episode_rec=[]
                    basline_rw_episode_tcn=[]

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            log.info("Updates {}, num timesteps {}, FPS {}  Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):

            vid_log_dir = os.getenv('TCN_ENV_VID_LOG_FOLDER', '/tmp/env_tcn/train_vid')
            vid_log_inter = os.getenv('TCN_ENV_VID_LOG_INTERVAL', '100')
            os.environ['TCN_ENV_VID_LOG_FOLDER'] ="eval_vid"# os.path.join(vid_log_dir,"../eval_vid/","interval_"+str(j))
            os.environ['TCN_ENV_VID_LOG_INTERVAL'] = '1'
            os.environ['TCN_ENV_EVAL_EPISODE']='1'
            with redirect_stdout(open(os.devnull, "w")):# no stdout
                with suppress_logging():
                    # eval envs
                    eval_envs = make_vec_envs(
                        args.env_name, args.seed + args.num_processes,1,
                        args.gamma, eval_log_dir, args.add_timestep, device, True)

                    vec_norm = get_vec_normalize(eval_envs)
                    if vec_norm is not None:
                        vec_norm.eval()
                        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

                    eval_episode_rewards = []

                    obs = eval_envs.reset()
                    eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                    actor_critic.recurrent_hidden_state_size, device=device)
                    eval_masks = torch.zeros(args.num_processes, 1, device=device)

                    while len(eval_episode_rewards) < 1:
                        with torch.no_grad():
                            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                        # Obser reward and next obs
                        obs, reward, done, infos = eval_envs.step(action)

                        eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                                   for done_ in done],
                                                   dtype=torch.float32,
                                                   device=device)

                        for info in infos:
                            if 'episode' in info.keys():
                                eval_episode_rewards.append(info['episode']['r'])

                    eval_envs.close()
            os.environ['TCN_ENV_VID_LOG_FOLDER'] = vid_log_dir
            os.environ['TCN_ENV_EVAL_EPISODE']='0'
            os.environ['TCN_ENV_VID_LOG_INTERVAL'] = vid_log_inter


            writer.add_scalar('eval/rw', np.mean(eval_episode_rewards), j)
            log.info(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

        if j % args.vis_interval == 0:
            try:
                td_plot(writer,args.log_dir)
                # Sometimes monitor doesn't properly flush the outputs
                # win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  # args.algo, args.num_env_steps)
            except IOError:
                print("plt error")
                pass


if __name__ == "__main__":
    main()
