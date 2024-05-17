import os
import subprocess
import numpy as np

GPU_FREE_SPACE_COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
GPU_NUMBER = None


def get_device_number(set_visible_devices=True):
    global GPU_NUMBER

    if GPU_NUMBER is None:
        output = subprocess.check_output(GPU_FREE_SPACE_COMMAND.split())
        free_spaces = list(map(int, output.decode().splitlines()))
        gpu_number = np.array(free_spaces).argmax()

        print("Found best GPU number:", gpu_number)
        print(f"\t(has {round(max(free_spaces) / 1024, 2)} GiB free)")

        if set_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
            print(f'Setting: os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_number}"')
        
        GPU_NUMBER = gpu_number

    return GPU_NUMBER
