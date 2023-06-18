# DASH: Decoupled Asynchronous Cloud Spot VM Checkpointing for Distributed Deep Learning

## What is DASH?

DASH is specifically designed for the Spot VM environment. DASH leverages the memory of a reliable remote VM as a checkpoint area, enabling asynchronous checkpointing. 

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
```

2. Initialize the DASH framework and obtain the necessary objects:

```python
communicator, train_node, remote_node = init_DASH(args, train_node_auto_start=True)
```

3. Use the train_node and remote_node objects for distributed training:

```python
# Perform distributed training using the train_node and remote_node objects
...

# Save model data to the remote node
train_node.save(model_data)
...

# Perform other operations
...
```

4. Destroy the DASH framework when training is completed:

```python3
destroy_DASH()
```

For more detailed information and example code, please refer to the example code provided in this repository.
