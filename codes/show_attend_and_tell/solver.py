import os
import torch
from net import Net
from dataset_CUB_1 import get_caption_dataset
#from visdomX import VisdomX
import mlflow
import numpy as np

class Solver():
    def __init__(self, args):
        self.train_loader, self.train_data, TEXT = get_caption_dataset(
            train=True,
            data_root=args.data_root,
            batch_size=args.batch_size, 
            image_size=args.image_size, 
            text_field=True)
        self.val_loader, _ = get_caption_dataset(
            train=False,
            data_root=args.data_root,
            batch_size=args.batch_size, 
            image_size=args.image_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = Net(TEXT, 
            args.hidden_dim, args.attn_dim, 
            args.num_layers).to(self.device)

        self.net.load_state_dict(torch.load("/home/abinaya/Documents/Show_and_tell_pytorch/codes/show_attend_and_tell/checkpoint/caption_20.pth"))

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1) # <pad>: 1
        self.optim   = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            args.lr)

        #self.vis = VisdomX()

        self.args = args

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def fit(self):
        args = self.args
        
        global_step = 0

        mlflow.set_tracking_uri('file:/home/abinaya/Documents/Show_and_tell_pytorch/codes/show_and_tell/mlruns')

        mlflow.set_experiment(experiment_name="Show and tell")

        with mlflow.start_run(run_name=args.mlflow_experiment_name):
            for epoch in range(args.max_epochs):
                self.net.train()
                val_epoch_loss = 0
                train_epoch_loss = 0

                for step, (inputs, test_inputs) in enumerate(zip(self.train_loader, self.val_loader)):
                    image   = inputs[0].to(self.device)
                    caption = inputs[1].to(self.device)
                    lengths = inputs[2].to(self.device)

                    val_image   = test_inputs[0].to(self.device)
                    val_caption = test_inputs[1].to(self.device)
                    val_lengths = test_inputs[2].to(self.device)
                    
                    # e.g.
                    # input: <start> this is caption
                    # gt:    this is caption <end>
                    #gt  = caption[:, 1:].contiguous().view(-1)

                    #out = self.net(image, caption[:, :-1], lengths-1)

                    out = self.net(image, caption, lengths)
                    loss = self.loss_fn(out, caption.view(-1))

                    #loss = self.loss_fn(out, gt)

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    train_epoch_loss += loss.item()

                    with torch.no_grad():
                        val_out = self.net(val_image, val_caption, val_lengths)
                        val_loss = self.loss_fn(val_out, val_caption.view(-1))
                        val_epoch_loss += val_loss.item()

                train_epoch_loss = train_epoch_loss / (step+1) 
                val_epoch_loss = val_epoch_loss / (step+1)
                mlflow.log_metric("Train loss", train_epoch_loss, step=epoch)
                mlflow.log_metric("Val loss", val_epoch_loss, step=epoch)

                if (epoch+1) % args.print_every == 0:
                    perplexity = np.exp(train_epoch_loss)
                    mlflow.log_metric("Train Perplexity", perplexity)
                    mlflow.log_metric("Val Perplexity", np.exp(val_epoch_loss))
                    #self.vis.add_scalars(perplexity, global_step+1,
                    #                    title="Attention-Perplexity",
                    #                    ylabel="Perplexity", xlabel="step")

                    print("Epoch [{}/{}] Epoch: [{}/{}] Train Perplexity: {:5.3f} Val Perplexity: {:5.3f}"
                        .format(epoch+1, args.max_epochs, 
                                int((epoch+1)), 
                                int(args.max_epochs*len(self.train_loader)),
                                perplexity, np.exp(val_epoch_loss)))

                if (epoch+1) % args.ckpt_every == 0:
                    self.save(args.ckpt_dir, args.ckpt_name, epoch+1)


    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
