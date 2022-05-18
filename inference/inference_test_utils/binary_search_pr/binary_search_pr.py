import os
import ast
import sys
import argparse
import subprocess
import numpy as np


def parse_args():
    """
    Parse the args of command line
    :return: all the args that user defined
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_commit',
        type=str,
        default='36d76840acf06b6b7f95803001dce9952cc43b77',
        help='the start pr of searching, it is a correct pr.'
    )
    parser.add_argument(
        '--end_commit',
        type=str,
        default='ce5e119696084cf8836a182df1b814c2dd80a256',
        help='the end pr of searching, it is a pr with problem.'
    )
    parser.add_argument(
        '--command',
        type=str,
        default='bash run.sh',
        help='command of running the script.'
    )
    parser.add_argument(
        '--cmake_command',
        type=str,
        default='cmake .. -DON_INFER=ON -DWITH_PYTHON=ON -DPY_VERSION=3.8 -DCMAKE_BUILD_TYPE=Release  \
                -DWITH_MKL=ON -DWITH_AVX=OFF -DWITH_MKLDNN=ON -DWITH_GPU=ON -DWITH_TENSORRT=ON  \
                -DTENSORRT_ROOT=/usr/local/TensorRT-8.0.3.4 -DWITH_TESTING=OFF -DWITH_INFERENCE_API_TEST=OFF  \
                -DWITH_DISTRIBUTE=OFF -DWITH_STRIP=ON -DWITH_CINN=OFF -DWITH_ONNXRUNTIME=OFF  \
                -DCUDA_ARCH_NAME=Auto',
        help='cmake command'
    )
    args = parser.parse_args()
    return args


def compile(commit_id):
    """
    compile paddle
    """
    uninstall_command = 'pip uninstall paddlepaddle-gpu -y'
    uninstall_res = os.system(uninstall_command)
    if uninstall_res == 0:
        print("uninstall successfully")
    else:
        print("uninstall failed")
        sys.exit(-1)
    reset_command = 'git reset --hard %s' % commit_id
    os.system(reset_command)
    command = 'rm -rf build && mkdir build'
    os.system(command)
    build_path = os.path.join(paddle_path, 'build')
    os.chdir(build_path)
    print('commit {} build start'.format(commit_id))
    cmake_command = args.cmake_command
    build_command = 'make -j32'
    os.system(cmake_command)
    build_res = os.system(build_command)
    if build_res == 0:
        print('commit {} build done'.format(commit_id))
    else:
        print('commit {} build failed'.format(commit_id))
        sys.exit(-1)
    install_command = 'pip install -U python/dist/paddlepaddle*'
    install_res = os.system(install_command)
    if install_res == 0:
        print("install paddle successfully")
    else:
        print("install paddle failed")
        sys.exit(-1)
    print('commit {} install done'.format(commit_id))
    return 0


def save(commit_id):
    """
    save paddle_wheel、inference_tar and logcd pa
    """
    # dir named commit_id is used to save paddle_wheel and model_log
    mkdir_cmd = 'mkdir {}'.format(commit_id)
    os.system(mkdir_cmd)
    mv_out_cmd = 'mv log.txt {}'.format(commit_id)
    mv_wheel_cmd = 'mv paddle/build/python/dist/paddlepaddle* {}'.format(commit_id)
    mv_tar_cmd = 'mv paddle/build/paddle_inference_install_dir {}'.format(commit_id)
    os.system(mv_wheel_cmd)
    os.system(mv_tar_cmd)
    os.system(mv_out_cmd)


def check_success(commit_id):
    """
    check result of specific commit.
    """
    os.chdir(paddle_path)
    compile(commit_id)
    os.chdir(base_path)
    print('base_path:{}'.format(base_path))

    cmd = args.command
    log_result = subprocess.getstatusoutput(cmd)
    exit_code = log_result[0]
    print(log_result)
    print('log:{}'.format(log_result[1]))
    with open('log.txt', 'w') as f:
        f.writelines(log_result[1])
    save(commit_id)
    if exit_code != 0:
        return False
    else:
        return True


def get_commits(start, end):
    """
    get all the commits in search interval
    """
    print('start:{}'.format(start))
    print('end:{}'.format(end))
    cmd = 'git log {}..{} --pretty=oneline'.format(start, end)
    log = subprocess.getstatusoutput(cmd)
    print(log[1])
    commit_list = []
    candidate_commit = log[1].split('\n')
    print(candidate_commit)
    for commit in candidate_commit:
        commit = commit.split(' ')[0]
        print('commit:{}'.format(commit))
        commit_list.append(commit)
    return commit_list


def binary_search(commits):
    """
    binary search
    """
    if len(commits) <= 2:
        if len(commits) == 1:
            res = commits[0]
        else:
            print('only two candidate commits left in binary_search: {}'.format(commits))
            if check_success(commits[1]):
                res = commits[0]
            else:
                res = commits[1]
        print('the final commit is {}'.format(res))
        return res
    left, right = 0, len(commits) - 1
    if left <= right:
        mid = left + (right - left) // 2
        commit = commits[mid]
        if check_success(commit):
            print('the commit {} is success'.format(commit))
            print('mid value:{}'.format(mid))
            selected_commits = commits[:mid]
            res = binary_search(selected_commits)
        else:
            print('the commit {} is failed'.format(commit))
            selected_commits = commits[mid:]
            res = binary_search(selected_commits)
    return res


if __name__ == '__main__':
    args = parse_args()
    base_path = os.getcwd()
    paddle_path = os.path.join(base_path, 'paddle')
    cmd = 'rm -rf paddle && git clone http://github.com/paddlepaddle/paddle.git'
    os.system(cmd)
    os.chdir(paddle_path)
    commits = get_commits(start=args.start_commit, end=args.end_commit)
    print('the candidate commits is {}'.format(commits))
    final_commit = binary_search(commits)
    print('the pr with problem is {}'.format(final_commit))
    with open('final_commit.txt', 'w') as f:
        f.writelines('the final commit is:{}'.format(final_commit))
