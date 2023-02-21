from trainner import Trainer_synapse
from config import FAUnet_synapse as hyper1
from config import FAUnet_synapse_pretrain as pretrain1
from config import FAUnet_synapse_pretrain2 as pretrain2
from config import FAUnet_synapse_pretrain3 as pretrain3


net = Trainer_synapse("D:\\Projects\\datasets\\Synapse_npy", pretrain3)


if __name__ == '__main__':
    # net.train()
    pass