import pdb
import subprocess
from datetime import datetime
import os


def call_training(task_id=1, result_dir='result_log/'):
    # os.makedirs(result_dir + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + "/")  # , exist_ok=True)

    run = subprocess.call(['python', 'single_char.py',
                           '--task_id', str(task_id)])


def find_param(type):

    global param_dict
    param_dict = {}
    tasks=[]
    if type=='rename':
        tasks=[1,2,3,6,7,8,9,11,12,13]
    if type=='replace':
        tasks =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,20]
    if type=='all':
        tasks=[1,2,3,6,7,8,9,11,12,13]
    if type=='babi':
        tasks=range(1,21,1)
    for task_id in tasks:#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,20]:#[1,2,3,6,7,8,9,11,12,13]:
        # pdb.set_trace()
        param_dict['task_id'] = task_id
        call_training(param_dict['task_id'])

def main():
    find_param('replace')

if __name__ == '__main__':
    main()