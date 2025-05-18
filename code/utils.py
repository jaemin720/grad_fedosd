import torch
import copy
from torchvision import datasets, transforms
import numpy as np


def get_dataset(args):
    """데이터셋 로드 및 사용자 그룹 분할 (IID / Non-IID)"""
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('../data/mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data/mnist', train=False, download=True, transform=transform)

    elif args.dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        train_dataset = datasets.FashionMNIST('../data/fashion_mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('../data/fashion_mnist', train=False, download=True, transform=transform)

    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform)

    else:
        raise ValueError(f'Unsupported dataset {args.dataset}')

    user_groups = partition_data(train_dataset, args)

    return train_dataset, test_dataset, user_groups


def partition_data(dataset, args):
    """IID or Non-IID 방식으로 데이터셋을 user 그룹별로 나누기"""
    num_items = int(len(dataset) / args.num_users)
    user_groups = {}

    if args.iid == 1:
        # IID: 데이터를 균등하게 섞어서 나누기
        idxs = np.random.permutation(len(dataset))
        for i in range(args.num_users):
            user_groups[i] = idxs[i * num_items:(i + 1) * num_items].tolist()
    else:
        # Non-IID: 임의로 샤딩하고 한 그룹에 특정 샤드 할당 (간단 예시)
        # 여기서는 MNIST 기준으로만 구현 (더 복잡한 방식은 논문 참조)
        # 실제 구현은 필요에 따라 수정할 것
        idxs = np.argsort(dataset.targets.numpy())
        shards_per_user = 2
        num_shards = args.num_users * shards_per_user
        shard_size = len(dataset) // num_shards
        shards = [idxs[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
        user_groups = {i: [] for i in range(args.num_users)}
        for i in range(args.num_users):
            assigned_shards = shards[i*shards_per_user:(i+1)*shards_per_user]
            for shard in assigned_shards:
                user_groups[i] += shard.tolist()

    return user_groups


def average_weights(w_list):
    """가중치 리스트를 평균내는 함수"""
    avg_weights = copy.deepcopy(w_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(w_list)):
            avg_weights[key] += w_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(w_list))
    return avg_weights


def exp_details(args):
    print(f'Experimental details:')
    print(f'    Model       : {args.model}')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Number of users : {args.num_users}')
    print(f'    Fraction of clients  : {args.frac}')
    print(f'    IID         : {bool(args.iid)}')
    print(f'    Local epochs: {args.local_ep}')
    print(f'    Local batch size: {args.local_bs}')
    print(f'    Learning rate: {args.lr}')
    print(f'    GPU enabled : {args.gpu}')
