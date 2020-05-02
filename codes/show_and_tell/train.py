import os
import argparse
from solver import Solver
import time


def main():
    base_dir = "/home/abinaya/Documents/Show_and_tell_pytorch/codes/show_and_tell/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=100)
    
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--data_root", type=str, default=base_dir+"data")
    parser.add_argument("--image_size", type=int, default=224)
    
    parser.add_argument("--ckpt_dir", type=str, default=base_dir+"checkpoint_resnet18_3DRNN_2")
    parser.add_argument("--ckpt_name", type=str, default="caption")
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--encoder_model", type=str, default="resnet-18") #renet18, resnet50
    parser.add_argument("--mlflow_experiment_name", type=str, default="Dilated LSTM decoder")
    
    args = parser.parse_args()
    solver = Solver(args)
    start = time.time()
    solver.fit()
    end = time.time()
    print('Time taken: {}', end-start)

if __name__ == "__main__":
    main()
