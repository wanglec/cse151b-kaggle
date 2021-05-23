import numpy as np
import pytorch_lightning as pl

# torch imports
import torch
from torch.utils.data import DataLoader

# internal imports
from src import config
from src.loss import RMSELoss
from src.model import TrajectoryModel
from src.data import ArgoverseDataset, collect_fn_train, collect_fn_test


class ArgoverseTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # determine if GPU exists
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # initialize model
        self.trajectory_model = TrajectoryModel(INPUT_SIZE)
        
    def forward(self, x):
        """Inference
        """
        decoder_outputs = self.trajectory_model(x)
        return decoder_outputs
    
    def training_step(self, batch, batch_ix):
        inp, out = batch
        inp = inp.to(self.device)
        out = out.to(self.device)

        # Set to train mode
        self.tracjectory_model.train()
        
        # Get relative position  
        initial_p_in = inp[:, 0, :2].detach().clone()
        inp[:, :, :2] = inp[:, :, :2] - initial_p_in[:, None]
        out[:, :, :2] = out[:, :, :2] - initial_p_in[:, None]
        
        # model training
        decoder_outputs = self.tracjectory_model(inp, out)
        
        # calculate loss
        loss = loss_fn(decoder_outputs, out.cpu())
        
        # log loss
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.tracjectory_model.parameters(), lr=config.LEARNING_RATE)
        return optimizer
    
    def validation_step(self, batch, batch_ix):
        pass
        

if __name__ == "__main__":
    # data
    train = ArgoverseDataset(data_path=config.TRAIN_PATH)
    test = ArgoverseDataset(data_path=config.TEST_PATH)
    
    train_size = int(config.TRAIN_SIZE * len(train))
    val_size = len(train) - train_size
    train, val = torch.utils.data.random_split(train, [train_size, val_size])
    
    train_loader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle = True, collate_fn=collect_fn_train, num_workers=0)
    val_loader = DataLoader(val, batch_size=config.BATCH_SIZE, shuffle = True, collate_fn=collect_fn_train, num_workers=0)
    test_loader = DataLoader(test, batch_size=config.BATCH_SIZE, shuffle = False, collate_fn=collect_fn_test, num_workers=0)
    
#     # model 
    model = ArgoverseTrainer()
    
#     # training
    trainer = pl.Trainer()
    trainer.fit(model, train_loader)
    
    
    
    
    