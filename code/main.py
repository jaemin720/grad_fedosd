#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import os
import time
import copy
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet
from utils import get_dataset, average_weights, exp_details


def select_model(args, train_dataset):
    """모델을 선택하고 초기화하는 함수"""
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            return CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            return CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            return CNNCifar(args=args)

    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        input_dim = np.prod(img_size)
        return MLP(dim_in=input_dim, dim_hidden=64, dim_out=args.num_classes)

    elif args.model == 'resnet':
        return ResNet(args=args)

    else:
        raise ValueError(f"Error: Unrecognized model {args.model}")


def main():
    start_time = time.time()

    args = args_parser()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    # 로그 저장 폴더 및 TensorBoard 설정
    os.makedirs('../logs', exist_ok=True)
    logger = SummaryWriter('../logs')

    exp_details(args)

    # 데이터셋, 사용자 그룹 불러오기
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # 모델 초기화 및 GPU 이동
    global_model = select_model(args, train_dataset).to(device)

    # 저장된 모델이 있으면 불러오기
    if args.load_model is not None and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        global_model.load_state_dict(torch.load(args.load_model, map_location=device))
        global_weights = global_model.state_dict()
    else:
        print("No pre-trained model loaded, training from scratch.")
        global_weights = global_model.state_dict()

    global_model.train()

    print(global_model)

    train_loss, train_accuracy = [], []

    print_every = 2

    for epoch in tqdm(range(args.epochs), desc='Global Training Rounds'):
        print(f'\n | Global Training Round : {epoch + 1} |')

        local_weights, local_losses = [], []

        global_model.train()

        m = max(int(args.frac * args.num_users), 1)
        selected_users = np.random.choice(range(args.num_users), m, replace=False)

        for user_idx in selected_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[user_idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)

        global_model.eval()
        accuracies = []
        for user_idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[user_idx], logger=logger)
            acc, _ = local_model.inference(global_model)
            accuracies.append(acc)
        avg_acc = sum(accuracies) / len(accuracies)
        train_accuracy.append(avg_acc)

        if (epoch + 1) % print_every == 0:
            print(f'\nAvg Training Stats after {epoch + 1} rounds:')
            print(f'Training Loss: {avg_loss:.4f}')
            print(f'Train Accuracy: {100 * avg_acc:.2f}%\n')

        logger.add_scalar('Loss/train', avg_loss, epoch)
        logger.add_scalar('Accuracy/train', avg_acc, epoch)

    # 테스트 데이터 평가
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f'\nResults after {args.epochs} global rounds of training:')
    print(f"|---- Avg Train Accuracy: {100 * train_accuracy[-1]:.2f}%")
    print(f"|---- Test Accuracy: {100 * test_acc:.2f}%")

    # 모델 저장
    if args.save_model is not None:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(global_model.state_dict(), args.save_model)
        print(f"Saved trained model to {args.save_model}")

    print(f'\nTotal Run Time: {time.time() - start_time:.4f} seconds')

    logger.close()


if __name__ == '__main__':
    main()
