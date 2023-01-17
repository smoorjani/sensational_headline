import argparse
import logging
USE_CUDA = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))

def get_args():
    parser = argparse.ArgumentParser(description="argument for nn parameters")


    parser.add_argument("--thd", type=float, default=0.0, help="thredhold for including samples to training")

    # deepspeed parameters
    parser.add_argument('--ds_config', type=str, default="ds_config_stage2.json", help="Deepspeed config file")
    parser.add_argument('--local_rank', type=int, default=0, help="local rank")

    ## nn parameters
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--hidden_size', type=int, default=768, help="hidden size")
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout rate")

    parser.add_argument('--generator', type=str, default="facebook/bart-base", help="load existing generator") 
    parser.add_argument('--discriminator_path', type=str, default="ckpt.pth", help="load existing discriminator") 
    parser.add_argument('--training_data', type=str, default="/control_tuning/dataset/train.txt", help="training data") 
    parser.add_argument('--eval_data', type=str, default="/control_tuning/dataset/eval.txt", help="eval data") 

    parser.add_argument('--save_path', type=str, default="/projects/bblr/smoorjani/control_tuning/models", help="save path") 
    
    ## optimization
    
    parser.add_argument('--epochs', type=int, default=10, help="epochs for runing")
    parser.add_argument('--total_steps', type=int, default=100000000, help="total steps for training")
    parser.add_argument('--optimizer', type=str, default="adam", help="which optimizer to use")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay for optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help="max grad norm")

    # hyperparams for loss
    parser.add_argument('--eps', type=float, default=1e-5, help="epison to avoid 0 probs")
    parser.add_argument('--gamma', type=float, default=0.1, help="gamma to weight speed MSE")

    # parser.add_argument("--use_rl", action='store_true', help="use rl or not")
    # parser.add_argument("--use_rep", action='store_true', help="use repetition loss or not")
    # parser.add_argument("--rl_lr", type=float, default=0.001, help="learning rate of rl")
    # parser.add_argument('--decay_lr', type=int, default=3, help="decay learning rate if validation is not improving")

     # parser.add_argument("--ml_wt", type=float, default=0.9, help="mle weight for combining")
    # parser.add_argument("--beta", type=float, default=0.1, help="maximum weight for sup loss")
    # parser.add_argument("--use_s_score", type=int, default=1, help="whether use sensation score or not in the rl training")


    ## other args
    args = parser.parse_args()
    assert args.thd is not None
    logging.info(args)

    return args



