#edited 2023/06/12

from ctypes import *
from typing import Union, Any
import pickle
import os
import threading
import queue
import time
from datetime import datetime
import io
import torch

class MPI:
    __mpi_module = CDLL('./mpi_module.so')
    __create_mpi_communication = __mpi_module.create_mpi_communication
    __create_mpi_communication.restype = c_void_p
    __delete_mpi_communication = __mpi_module.delete_mpi_communication
    __delete_mpi_communication.argtypes = [c_void_p]

    def __init__(self):
        self.__mpi_communication = self.__create_mpi_communication()

        self.__get_rank = self.__mpi_module.getRank
        self.__get_rank.argtypes = [c_void_p]
        self.__get_rank.restype = c_int
        self.rank = self.__get_rank (self.__mpi_communication)

        self.__get_size = self.__mpi_module.getSize
        self.__get_size.argtypes = [c_void_p]
        self.__get_size.restype = c_int
        self.size = self.__get_size (self.__mpi_communication)

        self.__get_processor_name = self.__mpi_module.getProcessorName
        self.__get_processor_name.argtypes = [c_void_p, POINTER(c_char_p), POINTER(c_int)]
        self.__get_processor_name.restype = None
        self.__processor_name = None

        self.__send = self.__mpi_module.send
        self.__send.argtypes = [c_void_p, c_int, c_int, c_void_p, c_int]
        self.__send.restype = None

        self.__recv = self.__mpi_module.recv
        self.__recv.argtypes = [c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_int)]
        self.__recv.restype = None

        self.__all_gather_int = self.__mpi_module.allGatherInt
        self.__all_gather_int.argtypes = [c_void_p, c_void_p, c_int, c_void_p, c_int]
        self.__all_gather_int.restype = None

        self.__free_buffer = self.__mpi_module.free_buffer
        self.__free_buffer.argtypes = [c_void_p]
        self.__free_buffer.restype = None

        self.sharding_rank_list = None

    def __del__(self):
        self.__delete_mpi_communication(self.__mpi_communication)

    def Get_processor_name(self):
        if self.__processor_name is None:
            data_buffer = c_char_p()
            data_buffer_size = c_int()
            self.__get_processor_name(self.__mpi_communication, byref(data_buffer), byref(data_buffer_size))
            self.__processor_name = data_buffer.value.decode() 
        return self.__processor_name

    def _send(self, data:Any, dest:int, tag:int) -> None:
        serialized_data = pickle.dumps(data)
        data_length = len(serialized_data)
        self.__send(self.__mpi_communication, c_int(int(dest)), c_int(int(tag)), serialized_data, data_length)

    def _byte_send(self, data:bytes, dest:int, tag:int) -> None:
        data_length = len(data)
        self.__send(self.__mpi_communication, c_int(int(dest)), c_int(int(tag)), data, data_length)

    def send(self, data:Any, dest:int, tag:int) -> None:
        if isinstance(data, bytes):
            self._byte_send(data, dest, tag)
        else:
            self._send(data, dest, tag)

    def _recv(self, source:int, tag:int) -> bytes:
        data_buffer = c_void_p()
        data_buffer_size = c_int()
        self.__recv(self.__mpi_communication, c_int(int(source)), c_int(int(tag)), byref(data_buffer), byref(data_buffer_size))
        recvdata = string_at(data_buffer, data_buffer_size.value)
        self.__free_buffer(data_buffer) # 0609추가2
        # del data_buffer # 0609추가1 # 두번째가설: data_buffer가 뭔가를 가지고있는가? 아님.
        return recvdata


    def recv(self, source:int, tag:int, deserialize=True) -> Union[bytes, Any]:
        if deserialize:
            return pickle.loads(self._recv(source, tag))
        return self._recv(source, tag)
    
    def makeShardingRank(self) -> int:
        array_type = c_int * self.size
        localRank = array_type ()
        my_local_rank = c_int(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))
        self.__all_gather_int(self.__mpi_communication, byref(my_local_rank), 1, localRank, 1)
        localRankList = [i for i in localRank]
        sharding_rank_list = []
        for i in range(self.size-1):
            if localRankList[i] == 0:
                sharding_rank_list.append(i)

        self.sharding_rank_list = sharding_rank_list
        return len(sharding_rank_list)

    def setEnviron(self, master_addr:str, master_port:str) -> None:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["RANK"] = f"{self.rank}"
        os.environ["WORLD_SIZE"] = f"{self.size-1}"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        
        if(int(os.environ["LOCAL_RANK"]) == 0):
            os.environ["SHARD_RANK"] = f"{self.sharding_rank_list.index(self.rank)}"
        else:
            os.environ["SHARD_RANK"] = "-1"

    def print(self, msg):
        print(f"rank {self.rank}: {msg}")

class COMPUTATION:
    def __init__(self, MPI: MPI, save_count:int, shard_rank: int, shard_size: int):
        self.__MPI = MPI
        self.__save_count = save_count
        self.sending_thread = threading.Thread(target=self.__send, args=())
        self.put_thread = threading.Thread(target=self.__put, args=())
        self.__job_queue = queue.Queue()
        self.model_data = None
        self.is_copy_complete = True
        self.copy_lock = threading.Lock()
        self.shard_rank = shard_rank
        self.shard_size = shard_size
    
    def start(self):
        if self.shard_rank>=0 and self.shard_rank<self.shard_size:
            self.sending_thread.start()
            self.put_thread.start()
        else:
            pass

    def __put(self):
        for _ in range(self.__save_count):
            while self.model_data is None:
                time.sleep(0.01)

            buffer = io.BytesIO()
            torch.save(self.model_data, buffer)
            self.model_data = None
            with self.copy_lock:
                self.is_copy_complete = True
            dump = buffer.getvalue()
            del buffer
            key_len = len(dump)
            
            compute_node_len = self.shard_size
            a = int(key_len / compute_node_len)
            r = int(key_len % compute_node_len)
            rank = self.shard_rank

            if rank+1 <= r:
                s = (rank)*(a+1)
                e = (rank+1)*(a+1)-1
            else:
                s = (rank)*a+r
                e = (rank+1)*a+r-1

            if s<=e:
                dump = dump[s:e+1]
                
            else:
                dump = bytes()

            self.__MPI.print(f"dump size: {len(dump)}")
            self.__MPI.print(f"#{self.shard_rank} @epoch {_+1}th serialization ends at {time.time()}") # 직렬화 끝.
            
            self.__job_queue.put(dump)


    def __send(self):
        for _ in range(self.__save_count):
            data = self.__job_queue.get()
            self.__MPI.send(data=data, dest=self.__MPI.size-1, tag=0)
            del data


class REMOTE:
    def __init__(self, MPI: MPI, save_count:int, data_buffer_size: int, shard_size: int, model_name: str="user_model"):
        self.__MPI = MPI
        self.__save_count = save_count
        self.__data_buffer_size = data_buffer_size
        self.__SHARD_SIZE = shard_size
        self.__model_name = model_name

        self.__greedy_flush_thread = threading.Thread(target=self.__greedy_flush, args=())
        self.__parallel_recv_thread = None
        self.__job_queue = queue.Queue()
        self.__dirty_bits = [0]*self.__data_buffer_size
        self.__dirty_bits_lock = threading.Lock()
        self.__data_buffer = [[0 for _ in range(self.__SHARD_SIZE)] for _ in range(self.__data_buffer_size)]
        self.__start_event = [threading.Event() for _ in range(self.__SHARD_SIZE)]
        self.__done_event = [threading.Event() for _ in range(self.__SHARD_SIZE)]
        self.__current_data_buffer_index = 0
        #self.__data_buffer_lock = [threading.Lock() for _ in range(self.__SHARD_SIZE)] #굳이필요없어보임
    
    def __getCurrentDataBufferIndex(self):
        return self.__current_data_buffer_index
    
    def __incCurrentDataBufferIndex(self):
        self.__current_data_buffer_index = (self.__current_data_buffer_index + 1) % self.__data_buffer_size

    def __greedy_flush(self):
        self.__MPI.print(f"save count. = {self.__save_count}")
        for count in range(self.__save_count):
            self.__MPI.print("greedy_flush : ready for get...")
            data_buffer_index = self.__job_queue.get()
            self.__MPI.print("greedy_flush : get!")
            
            # Solution 1
            # entire_recv_data = OrderedDict()
            # for i in range(self.__SHARD_SIZE):
            #     entire_recv_data.update(pickle.loads(self.__data_buffer[data_buffer_index][i]))
            # self.__MPI.print("greedy_flush : entire recv data is made.")

            # Solution 2
            entire_recv_data = b''.join(self.__data_buffer[data_buffer_index])
            self.__MPI.print(f"@epoch {count+1}th join ends at {time.time()}") # 직렬화 끝.
            with self.__dirty_bits_lock:
                self.__dirty_bits[data_buffer_index] = 0
            self.__MPI.print("greedy_flush : dirty-bit unset.")
            # file_path_name = f"/mnt/efs/{self.__model_name}/{self.__model_name}_{count+1}th_" + datetime.now().strftime("%Y-%m-%d-%H%M%S")+ ".pt.tar"
            file_path_name = f"{self.__model_name}.pt.tar"
            with open(file_path_name, 'wb') as f:
                f.write(entire_recv_data)
                f.flush()
                os.fsync(f.fileno()) # 파일 디스크립터 f.fileno를 넣어줌
                f.close() # with때문에 굳이 필요없음
            del entire_recv_data
            #debug
            #self.__MPI.print(type(pickle.loads(entire_recv_data)))
            # self.__MPI.print(f"{_+1}번째 file write 완료.")
            self.__MPI.print(f"@epoch {count+1}th save ends at {time.time()}")
        
        print(f"file size: {os.stat(file_path_name).st_size}")

    def __parallel_recv(self, rank_number: int):
        for _ in range(self.__save_count):
            self.__start_event[rank_number].wait()
            self.__start_event[rank_number].clear()
            data_buffer_index = self.__getCurrentDataBufferIndex()
            recv_data = self.__MPI.recv(source=self.__MPI.sharding_rank_list[rank_number], tag=0, deserialize=False)
            self.__MPI.print(f"#{rank_number} @epoch {_+1}th pgt ends at {time.time()}") # REmote Send 끝
            #self.__MPI.print(f"{rank_number} recved.")
            #with self.__data_buffer_lock[rank_number]:
            self.__data_buffer[data_buffer_index][rank_number] = recv_data
            self.__done_event[rank_number].set()

    def start(self):
        # if not os.path.exists("/mnt/efs/"+self.__model_name):
        #     os.makedirs("/mnt/efs/"+self.__model_name)

        self.__greedy_flush_thread.start()
        self.__parallel_recv_thread = [threading.Thread(target=self.__parallel_recv, args=(i,)) for i in range(self.__SHARD_SIZE)]

        for i in range(self.__SHARD_SIZE):
            self.__parallel_recv_thread[i].start()

        for _ in range(self.__save_count):
            cur = self.__getCurrentDataBufferIndex() # 써야하는 버퍼 번호

            while True: # 더티빗 0인거 확인하고
                if self.__dirty_bits[cur] == 0:
                    with self.__dirty_bits_lock:
                        self.__dirty_bits[cur] = 1 # 더티빗 셋
                        break

            for rank in range(self.__SHARD_SIZE): # recv 시킴
                self.__start_event[rank].set()
            
            for rank in range(self.__SHARD_SIZE): # recv 완료되었나 확인
                self.__done_event[rank].wait()
                self.__done_event[rank].clear()
                
            self.__job_queue.put(cur) # 다되면 lazy flush 시작

            self.__incCurrentDataBufferIndex() # 버퍼 숫자 증가
        
        for i in range(self.__SHARD_SIZE):
            self.__parallel_recv_thread[i].join()

        self.__greedy_flush_thread.join()
        
