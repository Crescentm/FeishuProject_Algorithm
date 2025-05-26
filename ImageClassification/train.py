# Train
from metric import calculate_metrics_macro, calculate_metrics_micro
from model import efficientnet_b7
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import argparse
from dataset import build_data_set
import os
from torch.utils.tensorboard import SummaryWriter
import tqdm




def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    train_dataset = build_data_set(args.image_size, args.train_data, is_train=True)  # 调用dataset.py中的build_data_set()方法
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valid_dataset = build_data_set(args.image_size, args.valid_data, is_train=False)  # 调用dataset.py中的build_data_set()方法
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    
    # 模型设计
    print("=> creating model '{}'".format(args.arch))
    model = efficientnet_b7(num_classes=args.classes_num).to(device)
    # model = timm.create_model('efficientnet_b2', pretrained=False, num_classes=3).to(device)
    # # 手动加载权重（适用于无法联网时）
    # state_dict = torch.load('weight/efficientnet_b2.pth', map_location=device)
    # model.load_state_dict(state_dict, strict=False)

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
    for epoch in range(args.epochs):  # args.epochs，epochs=10
        
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

        for i, (images, target) in enumerate(train_loader):   # 本案例中，训练图片是19张，所以train_loader是19/10,2个元素，每个元素10个大小
            images = images.cuda()  # 将数据放到GPU上
            target = target.cuda()  # 将数据放到GPU上
            # 验证下图片是否加载正确---------降个维度-------------Start
            # img_new=images[0,:,:,:]
            # display(transforms.ToPILImage()(img_new))
            # 验证下图片是否加载正确---------降个维度-------------END
            
            # print('第{}轮，第{}组，当前训练的一组数据大小是:{}'.format(epoch,i,images.shape))
            # print(a)
            
            # 计算输出
            output = model(images)
            # from pdb import set_trace
            # set_trace()
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
            # from pdb import set_trace as st
            # st()

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
                # print('第{}轮，第{}组，当前验证的一组数据大小是:{}'.format(epoch,i,images.shape))
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
                # from pdb import set_trace as st
                # st()
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
    parser.add_argument('--batch-size', default=32, dest='batch_size', type=int, help='batch size')
    parser.add_argument('--valid-batch-size', default=1, dest='valid_batch_size', type=int, help='validation batch size')
    parser.add_argument('--workers', default=4, dest='num_workers', type=int, help='worders number of Dataloader')
    parser.add_argument('--epochs', default=500, type=int, help='epochs')
    parser.add_argument('--lr',  default=0.0001,type=float, help='learning rate')
    parser.add_argument('--checkpoint-dir', default='/home/jh/Blog/code/ckpts_eff/', dest='checkpoint_dir', help='location of checkpoint')
    parser.add_argument('--save-interval', default=1, dest='save_interval', type=int, help='save interval')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay', help='weight decay')
    parser.add_argument('--arch', default='efficientnet-b7', help='arch type of EfficientNet')
    parser.add_argument('--pretrained', default=False, help='use pretrained model')
    parser.add_argument('--advprop', default=False, help='advprop')
    parser.add_argument('--classes_num', default=3,dest='classes_num', help='classes_num')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])

    # 设置随机种子
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark = True
    
    # 调用主函数
    main(args)