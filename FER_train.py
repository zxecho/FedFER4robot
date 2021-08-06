import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

from FedFER.function_utils import set_lr, clip_gradient, save_model, mkdir


# Training
def IdpTrain(args, IdpModel, name, dataset_loader, save_path):

    local_loss_list = []
    local_acc_list = []
    pre_acc = 0
    best_acc = 0

    Epochs = args.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(IdpModel.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.L2_weight_decay)
    # optimizer = torch.optim.Adam(IdpModel.parameters(), lr=args.lr)
    with tqdm(range(Epochs)) as tq:
        tq.set_description('Local {} Training Epoch: '.format(name))
        for epoch in tq:
            IdpModel.train()
            train_loss = 0
            correct = 0
            total = 0

            if epoch > args.decay_start >= 0:
                frac = (epoch - args.decay_start) // args.decay_every
                decay_factor = args.lr_decay ** frac
                current_lr = args.lr * decay_factor
                set_lr(optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = args.lr

            # print('Idpt learning_rate: %s' % str(current_lr))

            tq.set_postfix(Lr=current_lr)

            for batch_idx, (inputs, targets) in enumerate(dataset_loader):
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = IdpModel(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if args.grad_clip:
                    clip_gradient(optimizer, 0.1)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            pre_acc = torch.true_divide(correct, total)

            local_loss_list.append(train_loss / (batch_idx + 1))
            local_acc_list.append(pre_acc)

            # 保存模型的文件夹名称
            # file_name = save_pth_pre.split('/')[2]
            file_name = save_path.split('/')
            file_name = file_name[2] + '/' + file_name[3]
            if pre_acc > best_acc:
                best_acc = pre_acc
                save_model(IdpModel, file_name, name)

        # tqdm打印信息
        tq.set_postfix(Acc=best_acc)

        fig, axs = plt.subplots()
        axs.plot(range(len(local_loss_list)), local_loss_list, label='Training loss')
        axs.set_xlabel('Idpt training epochs')
        axs.plot(range(len(local_acc_list)), local_acc_list, label='Test accuracy')
        img_save_dir = save_path + 'idpt_training_info/'
        mkdir(img_save_dir)
        plt.legend()
        plt.savefig(img_save_dir+'Idpt{}_training_loss&acc.png'.format(name))
        plt.cla()
        plt.close()

    return best_acc


def LocalTrain(args, LocalModel, dataset_loader, client_numb, lr):
    Train_acc = 0

    Epochs = args.local_ep
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(LocalModel.parameters(), lr=lr, momentum=0.9, weight_decay=args.L2_weight_decay)
    Train_acc = 0

    criterion = nn.CrossEntropyLoss()
    local_loss_list = []
    local_acc_list = []
    with tqdm(range(Epochs)) as tq:
        tq.set_description('Local P {} Training Epoch: '.format(client_numb))
        for epoch in tq:
            # print('\n Current Epoch: %d' % epoch)
            LocalModel.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(dataset_loader):
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = LocalModel(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if args.grad_clip:
                    clip_gradient(optimizer, 0.1)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                # print("[>>>Local Training {}/{}]  ".format(batch_idx + 1, len(dataset_loader)),
                #       'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1),
                #                                             100. * correct / total, correct, total))
            local_loss_list.append(train_loss / (batch_idx + 1))

            epoch_acc = 100. * correct / total
            local_acc_list.append(torch.true_divide(correct, total).item())

            # tqdm打印信息
            tq.set_postfix(Acc=epoch_acc)

    Train_acc = 100. * correct / total

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(range(len(local_loss_list)), local_loss_list)
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Local training epochs')
    axs[1].plot(range(len(local_acc_list)), local_acc_list)
    axs[1].set_ylabel('Acc')
    axs[1].set_xlabel('Local training epochs')
    plt.savefig('./data/local_training_loss&acc.png')
    plt.cla()
    plt.close()

    return LocalModel.state_dict(), train_loss / (batch_idx + 1), dataset_loader.dataset.number


def Local_SCAFFOLD_Train(args, LocalModel, server_c, client_c, dataset_loader, client_numb, lr):

    global_model = copy.deepcopy(LocalModel)
    server_c_para = server_c.state_dict()
    c_local_para = client_c.state_dict()

    tau = 0
    Epochs = args.local_ep
    optimizer = torch.optim.SGD(LocalModel.parameters(), lr=lr, momentum=0.9, weight_decay=args.L2_weight_decay)

    criterion = nn.CrossEntropyLoss()
    local_loss_list = []
    local_acc_list = []
    with tqdm(range(Epochs)) as tq:
        tq.set_description('Local P {} Training Epoch: '.format(client_numb))
        for epoch in tq:
            # print('\n Current Epoch: %d' % epoch)
            LocalModel.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(dataset_loader):
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = LocalModel(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if args.grad_clip:
                    clip_gradient(optimizer, 0.1)
                optimizer.step()

                # control update direction
                net_para = LocalModel.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - lr * (server_c_para[key] - c_local_para[key])
                LocalModel.load_state_dict(net_para)

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                tau += 1

            local_loss_list.append(train_loss / (batch_idx + 1))

            epoch_acc = 100. * correct / total
            local_acc_list.append(torch.true_divide(correct, total).item())

            # tqdm打印信息
            tq.set_postfix(Acc=epoch_acc)

    c_new_para = client_c.state_dict()
    c_delta_para = copy.deepcopy(client_c.state_dict())
    global_model_para = global_model.state_dict()
    net_para = LocalModel.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - server_c_para[key] + (global_model_para[key] - net_para[key]) / (tau * lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    client_c.load_state_dict(c_new_para)

    Train_acc = 100. * correct / total

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(range(len(local_loss_list)), local_loss_list)
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Local training epochs')
    axs[1].plot(range(len(local_acc_list)), local_acc_list)
    axs[1].set_ylabel('Acc')
    axs[1].set_xlabel('Local training epochs')
    plt.savefig('./data/local_training_loss&acc.png')
    plt.cla()
    plt.close()

    return LocalModel.state_dict(), train_loss / (batch_idx + 1), len(dataset_loader), c_delta_para
