from trainner import Trainer_synapse
from config import FAUnet_synapse as hyper1
from config import FAUnet_synapse_pretrain as hyper2


net = Trainer_synapse("D:\\Projects\\datasets\\Synapse_npy", hyper2)


if __name__ == '__main__':
    net.train()