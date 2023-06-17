# DASH: Decoupled Asynchronous Cloud Spot VM Checkpointing for Distributed Deep Learning

## 0. What is DASH?
DASH is specifically designed for the Spot VM environment. DASH leverages the memory of a reliable remote VM as a checkpoint area, enabling asynchronous checkpointing. 

## 1. How to use
1. Compile mpi_module.cpp using the compilation shell.
2. Import DASH_package into the training code.
3. Refer to the example code and use DASH_package in the middle of the training code.
4. You can start distributed training using the learn shell.