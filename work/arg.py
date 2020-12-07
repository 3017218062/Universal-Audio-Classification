from __init__ import *

parser = argparse.ArgumentParser(description="Training")

parser.add_argument("-e", "--epochs", default=10, type=int)
parser.add_argument("-b", "--batch_size", default=32, type=int)
parser.add_argument("-a", "--accumulate_grad_batches", default=1, type=int)
parser.add_argument("-t", "--target_size", default=224, type=int)
parser.add_argument("-p", "--precision", default=16, type=int)
parser.add_argument("-m", "--model", default="dla60_res2next", type=str)
parser.add_argument("-l1", "--learning_rate", default=3e-4, type=float)
parser.add_argument("-l2", "--min_lr", default=1e-6, type=float)
parser.add_argument("-d", "--weight_decay", default=1e-6, type=float)
parser.add_argument("-f", "--fold", default=0, type=int)
parser.add_argument("-s", "--smooth", default=0., type=float)
parser.add_argument("-o", "--ohem", default=1., type=float)
parser.add_argument("-g", "--gpu", default=0, type=int)

args = parser.parse_args()
