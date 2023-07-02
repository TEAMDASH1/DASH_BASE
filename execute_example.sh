#!/bin/bash

# NP = # of TRAINING NODES + 1 (REMOTE NODE)
mpirun \
-np 5 \
-hostfile hosts \
python3 ./training_example.py \
10 1 172.31.12.34 1234 \
--batch_size 128 \
--remote_buffer_size 2 \
--model_name my_model \
--file_name_include_datetime True \
--file_save_in_dictionary True \
| tee my_model.result