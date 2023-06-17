from DASH_package import MPI, REMOTE, COMPUTATION

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group#, broadcast
import os
import time

MPI = MPI()

def ddp_setup (master_addr: str, master_port: str):
    MPI.setEnviron(master_addr, master_port)
    init_process_group (backend="nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    torch.cuda.set_device (int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__ (
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        save_every: int,
        shard_size: int,
        snapshot_path: str,     # CKP 정보.
        computation_node: COMPUTATION,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])    # gpu_id 대신: 한 머신 안에서.
        self.global_rank = int(os.environ["RANK"])        # gpu_id 대신: 여러 머신에 걸쳐서.
        self.shard_rank = int(os.environ["SHARD_RANK"])   # node를 대표하는 gpu
        
        self.model = model.to (self.local_rank)     # local_rank의 디바이스로 전송.
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.epochs_run = 1                     # CKP를 위함.
        self.snapshot_path = snapshot_path
        self.shard_size = shard_size
        self.computation_node = computation_node
        self.model = DDP (self.model, device_ids=[self.local_rank])     # [DDP] init 타임에 DDP class로 wrapping.
        if self.snapshot_path and os.path.exists (snapshot_path):
            # if self.global_rank == 0:
            print ("Loading snapshot")
            self._load_snapshot (snapshot_path)
            # broadcast(self.model.module.state_dict(), src=0)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)

        self.epochs_run = snapshot['epoch']+1
        self.model.module.load_state_dict(snapshot['model_state_dict'])
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        self.scheduler.load_state_dict(snapshot['scheduler_state_dict'])
        
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _make_snapshot (self, epoch) -> dict:
        snapshot = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict (),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        # torch.save (snapshot, self.snapshot_path)
        # print (f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        return snapshot

    def _run_batch (self, source, targets):
        self.optimizer.zero_grad ()
        output = self.model (source)
        loss = F.cross_entropy (output, targets)
        loss.backward ()
        self.optimizer.step ()
        return loss.item()

    def _run_batch_first_step(self, source, targets, epoch):
        self.optimizer.zero_grad ()
        output = self.model (source)
        loss = F.cross_entropy (output, targets)
        loss.backward ()
        stall_start = time.time()
        while not self.computation_node.is_copy_complete: # 첫번째 step 침범 방지.
            time.sleep(0.01) # unit time (이 시간보다 짧은 시간이라면 stall없다고 생각 가능.)
        MPI.print(f"#{self.shard_rank} @epoch {epoch}th cst ends at {(time.time() - stall_start):.10f}")
        self.optimizer.step ()
        return loss.item()
    
    def _run_batch_val (self, source, targets):
        self.optimizer.zero_grad ()
        output = self.model (source)
        loss = F.cross_entropy (output, targets)
        return loss.item()
    
    def _run_epoch_no_send (self, epoch):
        start_time = time.time()

        #train
        self.model.train()
        b_sz = len (next (iter (self.train_data))[0])
        train_loss = 0
        size = len(self.train_data)
        print_period = min(100, int(size/3))
        print (f"[GPU{self.global_rank}, X] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch (epoch)
        for batch, (source, targets) in enumerate(self.train_data):
            source = source.to (self.local_rank)        # local_rank 디바이스로 전송.
            targets = targets.to (self.local_rank)      # local_rank 디바이스로 전송.
            loss = self._run_batch (source, targets)
            train_loss += loss# * source.size(0)
            if batch % print_period == 0:
                MPI.print(f"train loss: {loss:>7f} [{batch*len(source):>5d}/{size*b_sz:5d}]")

        train_loss = train_loss / size
        train_end_time = time.time()

        #validation
        self.model.eval()
        b_sz = len (next (iter (self.validation_data))[0])
        val_loss = 0
        size = len(self.validation_data)
        with torch.no_grad():
            for batch, (source, targets) in enumerate(self.validation_data):
                source = source.to (self.local_rank)        # local_rank 디바이스로 전송.
                targets = targets.to (self.local_rank)      # local_rank 디바이스로 전송.
                loss = self._run_batch_val (source, targets)
                val_loss += loss
                # prediction = output.max (1, keepdim = True) [1]
                # correct += prediction.eq (targets.view_as (prediction)).sum ().item ()

        val_loss = val_loss / size
        self.scheduler.step(val_loss)
        val_end_time = time.time()

        # val_accuracy = 100. * correct / len (self.validation_data.dataset)
        MPI.print(f"{epoch} | n_duration = {val_end_time - start_time} | t_duration = {train_end_time - start_time} | v_duration = {val_end_time - train_end_time} | avg_train_loss = {train_loss} | avg_validation_loss = {val_loss}")# | validation_accuracy = {val_accuracy}")
        # MPI.print(f"{epoch} | n_duration = {train_end_time - start_time} | t_duration = {train_end_time - start_time} | avg_train_loss = {train_loss}")

    def _run_epoch (self, epoch):
        start_time = time.time()

        #train
        self.model.train()
        b_sz = len (next (iter (self.train_data))[0])
        train_loss = 0
        size = len(self.train_data)
        print_period = min(100, int(size/3))
        print (f"[GPU{self.global_rank}, {self.shard_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch (epoch)
        
        train_data_iter = iter(self.train_data)
        source, targets = next(train_data_iter)
        source = source.to (self.local_rank)        # local_rank 디바이스로 전송.
        targets = targets.to (self.local_rank)      # local_rank 디바이스로 전송.
        train_loss += self._run_batch_first_step (source, targets, epoch)
        MPI.print(f"train loss: {train_loss:>7f} [{0:>5d}/{size*b_sz:5d}]")

        for batch, (source, targets) in enumerate(train_data_iter, start=1):
            source = source.to (self.local_rank)        # local_rank 디바이스로 전송.
            targets = targets.to (self.local_rank)      # local_rank 디바이스로 전송.
            loss = self._run_batch (source, targets)
            train_loss += loss# * source.size(0)
            if batch % print_period == 0:
                MPI.print(f"train loss: {loss:>7f} [{batch*len(source):>5d}/{size*b_sz:5d}]")

        train_loss = train_loss / size
        train_end_time = time.time()
        
    #    #validation
    #     if MPI.rank == 0:
        self.model.eval()
        b_sz = len (next (iter (self.validation_data))[0])
        val_loss = 0
        size = len(self.validation_data)
        with torch.no_grad():
            for batch, (source, targets) in enumerate(self.validation_data):
                source = source.to (self.local_rank)        # local_rank 디바이스로 전송.
                targets = targets.to (self.local_rank)      # local_rank 디바이스로 전송.
                loss = self._run_batch_val (source, targets)
                val_loss += loss
                # prediction = output.max (1, keepdim = True) [1]
                # correct += prediction.eq (targets.view_as (prediction)).sum ().item ()

        val_loss = val_loss / size
        self.scheduler.step(val_loss)
        val_end_time = time.time()

        # val_accuracy = 100. * correct / len (self.validation_data.dataset)
        MPI.print(f"{epoch} | n_duration = {val_end_time - start_time} | t_duration = {train_end_time - start_time} | v_duration = {val_end_time - train_end_time} | avg_train_loss = {train_loss} | avg_validation_loss = {val_loss}")# | validation_accuracy = {val_accuracy}")
    #     else:
        # MPI.print(f"{epoch} | n_duration = {train_end_time - start_time} | t_duration = {train_end_time - start_time} | avg_train_loss = {train_loss}")

    def _train_no_send(self, max_epochs: int):
        for epoch in range (self.epochs_run, max_epochs+1):
            self._run_epoch_no_send (epoch)

    def _train_send(self, max_epochs: int):
        for epoch in range (self.epochs_run, max_epochs+1):
            self._run_epoch (epoch)
            if epoch % self.save_every == 0:
                #MPI.print(f"epoch={epoch}, save_every={self.save_every}")
                MPI.print(f"#{self.shard_rank} @epoch {epoch}th save starts at {time.time()}")
                with self.computation_node.copy_lock:
                    self.computation_node.is_copy_complete=False
                    self.computation_node.model_data = self._make_snapshot(epoch) # shallow copy - linking만 (pointing)
                MPI.print(f"#{self.shard_rank} @epoch {epoch}th shallow_copy ends at {time.time()}") # sallow copy 끝

    def train (self, max_epochs: int):
        if self.shard_rank == -1 or self.shard_rank >= self.shard_size: # 불가능 or sharding에 참여 x
            self._train_no_send(max_epochs)
        else:
            self._train_send(max_epochs)
    
class BasicNet (nn.Module):
    def __init__ (self):
        super ().__init__ ()
        #self.conv1 = nn.Conv2d (1, 32, 3, 1)   # MNIST.
        self.conv1 = nn.Conv2d (3, 32, 3, 1)    # CIFAR10.
        self.conv2 = nn.Conv2d (32, 64, 3, 1)
        self.dropout1 = nn.Dropout (0.25)
        self.dropout2 = nn.Dropout (0.5)
        #self.fc1 = nn.Linear (9216, 128)       # MNIST.
        self.fc1 = nn.Linear (12544, 128)       # CIFAR10.
        self.fc2 = nn.Linear (128, 10)
        self.act = F.relu

    def forward (self, x):
        x = self.act (self.conv1 (x))
        x = self.act (self.conv2 (x))
        x = F.max_pool2d (x, 2)
        x = self.dropout1 (x)
        x = torch.flatten (x, 1)
        x = self.act (self.fc1(x))
        x = self.dropout2 (x)
        x = self.fc2 (x)
        output = F.log_softmax (x, dim=1)
        return output

def load_train_objs ():
    #train_set = MyTrainDataset(2048)  # load your dataset

    transform = transforms.Compose ([
        transforms.ToTensor (),
        transforms.Normalize ((0.1307), (0.3081))
    ])    
    train_set = datasets.CIFAR10 ('data', train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10 ('data', train=False, download=True, transform=transform)

    # model = torch.nn.Linear(20, 1)  # load your model
    # model = BasicNet () # cifar10

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor (),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor (),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])

    # train_set = datasets.ImageNet('/mnt/efs/imagenet', split='train', transform=train_transform)
    # val_set = datasets.ImageNet('/mnt/efs/imagenet', split='val', transform=val_transform)
    
    # model = models.vgg16(num_classes=10)
    # model = models.vgg19(num_classes=10)
    # model = models.resnet18(num_classes=10)
    # model = models.resnet50(num_classes=10)
    model = models.resnet152(num_classes=10)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201')
    # model = models.efficientnet_v2_l (num_classes=10)
    # model = models.efficientnet_v2_l (num_classes=10)

    # optimizer = torch.optim.SGD (model.parameters(), lr=1e-1, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=False)

    return train_set, val_set, model, optimizer, scheduler

def prepare_dataloader (dataset: Dataset, batch_size: int, DDP:bool=True):
    if DDP:
        return DataLoader (
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler (dataset)
        )
    else:
        return DataLoader (
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
        )

def main (master_addr: str, master_port: str, save_every: int, total_epochs: int, batch_size: int, shard_size: int, snapshot_path: str):
    main_start = time.time ()
    ddp_setup (master_addr, master_port)
    train_dataset, validation_dataset, model, optimizer, scheduler = load_train_objs ()
    train_data = prepare_dataloader (train_dataset, batch_size)
    valldation_data = prepare_dataloader(validation_dataset, batch_size, False)
    computation_node = COMPUTATION(MPI, int(total_epochs/save_every), int(os.environ["SHARD_RANK"]), shard_size) # 0,1,2,3 - 4개, buffer, send관리 threads
    computation_node.start() # thread start.
    trainer = Trainer (model, train_data, valldation_data, optimizer, scheduler, save_every, shard_size, snapshot_path, computation_node)
    trainer.train (total_epochs)
    destroy_process_group ()
    
    main_end = time.time ()
    main_duration = main_end - main_start
    MPI.print (f'main_duration: {main_duration}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser (description='simple distributed training job')
    parser.add_argument ('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument ('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument ('master_addr', type=str, help='Master node IP address')
    parser.add_argument ('master_port', type=str, help="Master node port number")
    parser.add_argument ('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument ('--data_buffer_size', default=1, type=int, help='Input data buffer size of remote node. (default: 1)')
    parser.add_argument ('--shard_size', default=1, type=int, help='Input size of sharding (default: 1)')
    parser.add_argument ('--model_name', default='user_model', type=str, help='Input your ML model name (default: user_model)')
    parser.add_argument ('--snapshot_path', default=None, type=str, help='Input checkpoint file name (default: None)')
    args = parser.parse_args ()

    max_shard_size = MPI.makeShardingRank()
    assert args.shard_size <= max_shard_size, "The shard size is too large."

    if MPI.rank == MPI.size-1:
        remote_node = REMOTE(MPI, int(args.total_epochs/args.save_every), args.data_buffer_size, args.shard_size, args.model_name)
        remote_node.start() # remote start! (remote fork)

    else:
        if MPI.rank == 0:
            print(f"total_epochs: {args.total_epochs}")
            print(f"save_every: {args.save_every}")
            print(f"batch_size: {args.batch_size}")
            print(f"data_buffer_size: {args.data_buffer_size}")
            print(f"shard_size: {args.shard_size}")
            print(f"model_name: {args.model_name}")
        main(master_addr=args.master_addr, master_port=args.master_port,
             save_every=args.save_every, total_epochs=args.total_epochs,
             batch_size=args.batch_size, shard_size=args.shard_size, snapshot_path=args.snapshot_path)
