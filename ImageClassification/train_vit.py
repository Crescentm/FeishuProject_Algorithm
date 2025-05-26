# Train
from metric import calculate_metrics_macro, calculate_metrics_micro
import torch.nn as nn
import torch
import argparse
from dataset import build_data_set
import os
from torch.utils.tensorboard import SummaryWriter
import tqdm
import timm
from torch.utils.data import DataLoader
from torchvision import transforms


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    train_dataset = build_data_set(args.image_size, args.train_data, is_train=True)  # 训练集启用增强
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valid_dataset = build_data_set(args.image_size, args.valid_data, is_train=False)  # 验证集无增强
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 模型设计
    print("=> creating model '{}'".format(args.arch))
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.classes_num, img_size=args.image_size).to(device)
    
    # 手动加载预训练权重
    state_dict = torch.load('weight/vit_base_patch16_224.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # 冻结部分参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻后6层 Transformer Blocks、分类头、位置嵌入和类标记
    for i in range(6, 12):  # ViT-Base 有12层，解冻 blocks[6:12]
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    model.pos_embed.requires_grad = True  # 直接设置 pos_embed 的 requires_grad
    model.cls_token.requires_grad = True  # 直接设置 cls_token 的 requires_grad

    # 验证冻结情况（可选）
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad = {param.requires_grad}")

    # 损失函数选择
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 优化方法选择
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 学习率调整
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)  # 余弦退火学习率调整
    
    # 可视化训练
    writer = SummaryWriter()

    # 权重保存路径
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    best_F1 = 0.0
    
    # 正式训练 
    for epoch in range(args.epochs):  # args.epochs，epochs=50
        
        # switch to train mode
        model.train()
        # 训练
        train_f1_micro = 0.0
        train_recall_micro = 0.0
        train_precision_micro = 0.0

        train_f1_macro = 0.0
        train_recall_macro = 0.0
        train_precision_macro = 0.0

        tq = tqdm.tqdm(total=train_loader.__len__())
        tq.set_description('Train: Epoch {}, lr {:.5f}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        for i, (images, target) in enumerate(train_loader):
            images = images.cuda()  # 将数据放到GPU上
            target = target.cuda()  # 将数据放到GPU上
            
            # 计算输出
            output = model(images)
            # 计算loss
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # 计算Precision Recall和F1
            precision_micro, recall_micro, f1_micro = calculate_metrics_micro(output, target)
            precision_macro, recall_macro, f1_macro = calculate_metrics_macro(output, target)

            train_f1_micro += f1_micro
            train_recall_micro += recall_micro
            train_precision_micro += precision_micro

            train_f1_macro += f1_macro
            train_recall_macro += recall_macro
            train_precision_macro += precision_macro

            # 可视化
            train_step = epoch * len(train_loader) + i
            writer.add_scalar('Loss/train', loss.item(), train_step)
            tq.set_postfix(classify_cn_loss='{:.5f}'.format(loss.item()))
            tq.update()

        num_batches = i + 1
        train_f1_micro_mean = train_f1_micro / num_batches
        train_recall_micro_mean = train_recall_micro / num_batches
        train_precision_micro_mean = train_precision_micro / num_batches

        train_f1_macro_mean = train_f1_macro / num_batches
        train_recall_macro_mean = train_recall_macro / num_batches
        train_precision_macro_mean = train_precision_macro / num_batches
        
        # 可视化训练
        writer.add_scalar('Precision/train_micro', train_precision_micro_mean, epoch)
        writer.add_scalar('Recall/train_micro', train_recall_micro_mean, epoch)
        writer.add_scalar('F1/train_micro', train_f1_micro_mean, epoch)

        writer.add_scalar('Precision/train_macro', train_precision_macro_mean, epoch)
        writer.add_scalar('Recall/train_macro', train_recall_macro_mean, epoch)
        writer.add_scalar('F1/train_macro', train_f1_macro_mean, epoch)
        tq.close()

        # 验证
        model.eval()
        valid_f1_micro = 0.0
        valid_recall_micro = 0.0
        valid_precision_micro = 0.0

        valid_f1_macro = 0.0
        valid_recall_macro = 0.0
        valid_precision_macro = 0.0
        tq = tqdm.tqdm(total=valid_loader.__len__())
        tq.set_description('Valid: Epoch {}, lr {:.5f}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        with torch.no_grad():
            for i, (images, target) in enumerate(valid_loader):
                images = images.to(device)  # Move input to GPU
                target = target.to(device)  # Move target to GPU
                output = model(images)
                # 计算Precision Recall和F1
                precision_micro, recall_micro, f1_micro = calculate_metrics_micro(output, target)
                precision_macro, recall_macro, f1_macro = calculate_metrics_macro(output, target)

                valid_f1_micro += f1_micro
                valid_recall_micro += recall_micro
                valid_precision_micro += precision_micro

                valid_f1_macro += f1_macro
                valid_recall_macro += recall_macro
                valid_precision_macro += precision_macro
                tq.set_postfix(classify_cn_loss='{:.5f}'.format(loss.item()))
                tq.update()
            num_batches = i + 1
            valid_f1_micro_mean = valid_f1_micro / num_batches
            valid_recall_micro_mean = valid_recall_micro / num_batches
            valid_precision_micro_mean = valid_precision_micro / num_batches

            valid_f1_macro_mean = valid_f1_macro / num_batches
            valid_recall_macro_mean = valid_recall_macro / num_batches
            valid_precision_macro_mean = valid_precision_macro / num_batches
            tq.close()
        print('Finish Epoch {}...'.format(epoch + 1))
        # 可视化
        writer.add_scalar('Precision/valid_micro', valid_precision_micro_mean, epoch)
        writer.add_scalar('Recall/valid_micro', valid_recall_micro_mean, epoch)
        writer.add_scalar('F1/valid_micro', valid_f1_micro_mean, epoch)
        
        writer.add_scalar('Precision/valid_macro', valid_precision_macro_mean, epoch)
        writer.add_scalar('Recall/valid_macro', valid_recall_macro_mean, epoch)
        writer.add_scalar('F1/valid_macro', valid_f1_macro_mean, epoch)
        # 保存最好的模型
        if min(valid_f1_micro_mean, valid_f1_macro_mean) > best_F1:
            best_F1 = min(valid_f1_micro_mean, valid_f1_macro_mean)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print('Best model saved with F1: {:.5f}'.format(best_F1))
    scheduler.step()  # 更新学习率
    writer.close()  # Close the writer after training is done


if __name__ == '__main__':
    # 实例化一个参数对象
    parser = argparse.ArgumentParser(description="---------------- 图像分类Sample -----------------")   
    # 下面开始正式的加载参数：别名key，及对应的值value
    parser.add_argument('--train-data', default='/home/jh/Blog/dataset/train', dest='train_data', help='location of train data')
    parser.add_argument('--valid-data', default='/home/jh/Blog/dataset/val', dest='valid_data', help='location of validation data')
    parser.add_argument('--image-size', default=256, dest='image_size', type=int, help='size of input image')
    parser.add_argument('--batch-size', default=128, dest='batch_size', type=int, help='batch size')
    parser.add_argument('--valid-batch-size', default=1, dest='valid_batch_size', type=int, help='validation batch size')
    parser.add_argument('--workers', default=4, dest='num_workers', type=int, help='workers number of Dataloader')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--checkpoint-dir', default='/home/jh/Blog/code/ckpts/', dest='checkpoint_dir', help='location of checkpoint')
    parser.add_argument('--save-interval', default=1, dest='save_interval', type=int, help='save interval')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-3, type=float, dest='weight_decay', help='weight decay')
    parser.add_argument('--arch', default='vit_base_patch16_224', help='arch type of Vision Transformer')
    parser.add_argument('--pretrained', default=True, help='use pretrained model')
    parser.add_argument('--advprop', default=False, help='advprop')
    parser.add_argument('--classes_num', default=3, dest='classes_num', help='classes_num')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark = True
    
    # 调用主函数
    main(args)