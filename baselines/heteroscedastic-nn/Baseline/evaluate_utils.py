import numpy as np
import os
from datetime import datetime

def get_RMSE(list1, list2):
    assert len(list1) == len(list2)
    sum = 0
    for i in range(len(list1)):
        sum += (list1[i] - list2[i])**2
    result = np.sqrt(sum / len(list1))
    
    return result

def get_RMAE(list1, list2):
    assert len(list1) == len(list2)
    sum = 0
    for i in range(len(list1)):
        sum += np.abs(list1[i] - list2[i])
    result = sum / len(list1)
    
    return result



def save_variable_to_file(variable, path):
    # Ensure the path exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Format the current time to append to the file name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{current_time}.txt"
    full_path = os.path.join(path, file_name)
    
    # Save the variable to the file
    with open(full_path, "w") as file:
        file.write(str(variable))
    
    return full_path
