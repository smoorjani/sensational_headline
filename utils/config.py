import argparse
import logging
USE_CUDA = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))

def get_args():
    parser = argparse.ArgumentParser(description="argument for nn parameters")
    parser.add_argument("--ml_wt", type=float, default=0.0, help="mle weight for combining")
    parser.add_argument("--use_s_score", type=int, default=None, help="whether use sensation score or not in the rl training")
    parser.add_argument("--thd", type=float, default=None, help="thredhold for training")

    # seq2seq parameters
    ## nn parameters
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--hidden_size', type=int, default=768, help="hidden size")
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout rate")

    parser.add_argument("--use_rl", type=bool, default=True, help="use rl or not")
    parser.add_argument("--rl_lr", type=float, default=0.001, help="learning rate of rl")

    parser.add_argument('--persuasivness_clasifier_path', type=str, default="persuasive_model.pt", help="load existing persuasiveness model") 
    parser.add_argument('--training_data', type=str, default="persuasive_pairs_train.txt", help="training data") 
    parser.add_argument('--eval_data', type=str, default="persuasive_pairs_test.txt", help="eval data") 
    
    ## optimization
    parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('-decay_lr', type=int, default=3, help="decay learning rate if validation is not improving")
    parser.add_argument('-epochs', type=int, default=100, help="epochs for runing")
    parser.add_argument('-total_steps', type=int, default=100000000, help="total steps for training")
    parser.add_argument('-optimizer', type=str, default="adam", help="which optimizer to use")
    parser.add_argument('-max_grad_norm', type=float, default=2.0, help="max grad norm")
    parser.add_argument('-eps', type=float, default=1e-12, help="epison to avoid 0 probs")

    ## other args
    args = vars(parser.parse_args())
    assert args["use_s_score"] is not None
    assert args["thd"] is not None
    args["use_s_score"] = bool(args["use_s_score"])
    logging.info(args)


