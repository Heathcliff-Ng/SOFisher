import argparse

def get_dqn_args():
    parser = argparse.ArgumentParser(description='PyTorch DQN example')
    parser.add_argument('--rs', type=int, default=200, metavar='N')
    parser.add_argument('--grid_x', type=int, default=5, metavar='N')
    parser.add_argument('--grid_y', type=int, default=5, metavar='N')
    parser.add_argument('--cell_num', type=int, default=13, metavar='N')
    parser.add_argument('--agent_name', default="DQN",
                        help='name of the environment to run, including DQN')
    parser.add_argument('--max_steps', type=int, default=5000000, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--evaluate_times', type=int, default=1, metavar='N',
                        help='Evaluates a policy for an average over 10 times (default: 10)')
    parser.add_argument('--eval_freq', type=int, default=1000, metavar='N',
                        help='Evaluates a policy every 10000 steps (default: 10000)')
    parser.add_argument('--fig_step', type=int, default=500000, metavar='N')
    parser.add_argument('--fig_freq', type=int, default=100000, metavar='N')
    parser.add_argument('--env_noisy_scale', type=float, default=0., metavar='G',
                        help='the scale of noisy added to the Env')
    parser.add_argument('--env_noisy_type', default="sin",
                        help='the type of noisy added to the Env')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 456)')
    parser.add_argument('--episode_size', type=int, default=50, metavar='N',
                        help='Maximum steps in each episode (default: 1000)')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DQN/')
    parser.add_argument('--reward_path', type=str, default='./output_images/avg_reward.png')
    parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')
    parser.add_argument('--max_size', type=int, default=100000, metavar='N',
                        help='maximum number of memory size (default: 1000000)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G')
    parser.add_argument('--epsilon', type=float, default=1.0, metavar='G')
    parser.add_argument('--eps_end', type=float, default=0.01, metavar='G')
    parser.add_argument('--eps_dec', type=float, default=5e-4, metavar='G')
    parser.add_argument('--fc1_dim', type=int, default=256, metavar='N')
    parser.add_argument('--fc2_dim', type=int, default=256, metavar='N')

    args, unknown = parser.parse_known_args()

    return args



