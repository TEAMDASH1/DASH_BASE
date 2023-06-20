# DASH: Decoupled Asynchronous Cloud Spot VM Checkpointing for Distributed Deep Learning

## What is DASH?

DASH is specifically designed for the Spot VM environment. DASH leverages the memory of a reliable remote VM as a checkpoint area, enabling asynchronous checkpointing. 

> **Note:** This GitHub repository is created for anonymous submission purposes as part of the IISWC 2023 research paper.

## Key Features

- Multilevel checkpointing: Enables the storage and retrieval of intermediate model data during distributed training.
- Remote node abstraction: Abstracts the remote (peer) node responsible for receiving and storing data from training nodes.
- Efficient data management: Implements circular buffers and dirty bits to efficiently manage and flush data from remote nodes.
- Parallel processing: Utilizes multithreading and parallel execution to optimize data transfer and storage operations.
- Simple integration: Provides easy-to-use functions for initializing DASH nodes and integrating them into existing training pipelines.

## Getting Started

### Installation

To use DASH, clone the repository from GitHub:

```shell
git clone https://github.com/TEAMDASH1/DASH.git
```

### Usage

To use DASH in your training script, follow these steps:

1. Import the necessary modules:

```python
import DASH
from torch.nn.parallel import DistributedDataParallel as DDP
```

2. Initialize the DASH framework and obtain the necessary objects:

```python
args = parser.parse_args ()
communicator, train_node, remote_node = DASH.init_DASH(args)
```

3. Use the train_node object for distributed training:

```python
# Create the model
model = ...

# Wrap the model with PyTorch DDP
model = DDP(model)

# Perform distributed training using the train_node object
...
train.node.wait_copy_complete()
optimizer.step()
...

# Save model data to the remote node
train_node.save(model.state_dict()) # Option 1
train_node.save({
    'model_state_dict': self.model.module.state_dict (),
    ...
}) # Option 2
...

# Perform other operations with the train_node object
...
```
> Note 1: Before updating your model using **optimizer.step()**, make sure to call **wait_copy_complete()**.

> Note 2: The remote_node object is responsible for receiving and storing data from training nodes and does not require user control.

4. Destroy the DASH framework when training is completed:

```python3
DASH.destroy_DASH()
```

For more detailed information and example code, please refer to the example code provided in this repository.
