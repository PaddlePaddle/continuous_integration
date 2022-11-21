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
        '--type',
        type=str,
        default='frame',
        help='frame or model, default is frame'
    )
    parser.add_argument(
        '--chain',
        type=str,
        default='',
        help='the chain name'
    )
    parser.add_argument(
        '--repo',
        type=str,
        default='',
        help='repo name'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='',
        help='the model name'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='model repo path for to get commit_id, it must be a absolute path'
    )
    parser.add_argument(
        '--frame_path',
        type=str,
        default='',
        help='frame repo path for to get commit_id, it must be a absolute path'
    )
    parser.add_argument(
        '--frame_commit',
        type=str,
        default='',
        help=''
    )
    parser.add_argument(
        '--model_commit',
        type=str,
        default='',
        help=''
    )
    parser.add_argument(
        '--start_commit',
        type=str,
        default='40f54537256c4780593064191c1c1a3d5409d4cc',
        help='the start pr of searching, it is a correct pr.'
    )
    parser.add_argument(
        '--end_commit',
        type=str,
        default='ac2e2e6b7f8e4fa449c824ac9f4d23e3af05c7d3',
        help='the end pr of searching, it is a pr with problem.'
    )
    parser.add_argument(
        '--frame_branch',
        type=str,
        default='',
        help=''
    )
    parser.add_argument(
        '--docker_image',
        type=str,
        default='',
        help=''
    )
    parser.add_argument(
        '--code_bos',
        type=str,
        default='',
        help=''
    )
    parser.add_argument(
        '--sender',
        type=str,
        default='',
        help=''
    )
    parser.add_argument(
        '--reciver',
        type=str,
        default='',
        help=''
    )
    parser.add_argument(
        '--mail_proxy',
        type=str,
        default='',
        help=''
    )
    args = parser.parse_args()
    return args


def install_paddle(commit_id):
    """
    install paddle
    """
    uninstall_command = 'pip uninstall paddlepaddle-gpu -y'
    uninstall_res = os.system(uninstall_command)
    if uninstall_res == 0:
        print("uninstall successfully")
    else:
        print("uninstall failed")
        sys.exit(-1)
    #os.environ['no_proxy'] = 'bcebos.com'
    install_command = 'pip install -U https://paddle-qa.bj.bcebos.com/paddle-pipe' \
                      'line/Develop-GpuAll-LinuxCentos-Gcc82-Cuda102-Trtoff-Py37-Compile/{}/paddle' \
                      'paddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl'.format(commit_id)
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
    save paddle_wheel、model_output and log 
    """
    # dir named commit_id is used to save paddle_wheel and model_log
    mkdir_cmd = 'mkdir {}'.format(commit_id)
    os.system(mkdir_cmd)
    mv_out_cmd = 'mv RESULT RESULT_{}; cd test_tipc; mv output {}; cd -'.format(commit_id, commit_id)
    os.system(mv_out_cmd)


def check_success(commit_id, args):
    """
    check result of specific commit.
    """
    if args.type == "model":
        cmd = "bash debug.sh {} {} {} {} {} {} {} {} {} {} {} {}".format(
               args.model_name,
               commit_id,
               args.frame_commit,
               args.repo,
               args.chain,
               args.frame_branch,
               args.docker_image,
               args.code_bos,
               args.sender,
               args.reciver,
               args.mail_proxy,
               args.frame_path)
    else:
        cmd = "bash debug.sh {} {} {} {} {} {} {} {} {} {} {} {}".format(
               args.model_name,
               args.model_commit,
               commit_id,
               args.repo,
               args.chain,
               args.frame_branch,
               args.docker_image,
               args.code_bos,
               args.sender,
               args.reciver,
               args.mail_proxy,
               args.frame_path)
    print(cmd)
    cmd_res = os.system(cmd)
    #save(commit_id)
    if cmd_res == 0:
        return True
    else:
        return False


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
    #print(candidate_commit)
    for commit in candidate_commit:
        commit = commit.split(' ')[0]
        #print('commit:{}'.format(commit))
        commit_list.append(commit)
    return commit_list


def binary_search(commits, args):
    """
    binary search
    """
    if len(commits) <= 2:
        print('only two candidate commits left in binary_search, the final commit is {}'.format(commits[0]))
        return commits[0]
    left, right = 0, len(commits) - 1
    if left <= right:
        mid = left + (right - left) // 2
        commit = commits[mid]
        if check_success(commit, args):
            print('the commit {} is success'.format(commit))
            # right = mid
            print('mid value:{}'.format(mid))
            selected_commits = commits[:mid + 1]
            res = binary_search(selected_commits, args)
        else:
            print('the commit {} is failed'.format(commit))
            # left = mid
            selected_commits = commits[mid:]
            res = binary_search(selected_commits, args)
    return res


def run():
    """
    """
    args = parse_args()

    work_path = os.getcwd()
    if args.type == "frame":
        os.chdir(args.frame_path)
    else:
        os.chdir(args.model_path)
    commits = get_commits(start=args.start_commit, end=args.end_commit)
    print(work_path)
    print(len(commits), commits)
    os.chdir(work_path)

    final_commit = binary_search(commits, args)
    print("~"*50)
    print("type:", args.type)
    print("repo:", args.repo)
    print("commit:", final_commit)
    print("~"*50)

    f = open('final_commit.txt', 'w')
    f.writelines('the final commit is:{}'.format(final_commit))
    f.close()


if __name__ == '__main__':
    run()

