class Args:
    data_root = r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset'
    train_list = r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset\train_small.txt'
    val_list = r'd:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset\test_small.txt'
    arch = 'resnet50' # 网络架构, 支持: resnet50, se_resnet50, efficientnet_b0, efficientnet_b1, efficientnet_b2, mobilenet_v3_small, mobilenet_v3_large, resnext50_32x4d, resnext101_32x8d
    num_classes = 9 # 类别数 (当前dataset目录下有9种缺陷类别：0_scratch, 1_broken, 1_gline, 2_bubble, 2_crack, 3_defect, 3_dent, 4_discolor, 4_unformed, 5_foreign_matter, 5_pinhole, 6_burr, 7_foreign, 7_lf, 8_pin，但实际训练时使用的是9个类别标签)
    batch_size = 32 # 减小批次大小，有助于模型泛化
    lr = 0.0001 # 降低学习率，微调预训练模型需要更小的LR
    momentum = 0.9
    weight_decay = 1e-5
    warm_up = 100 # lr warm_up step
    epoch = 50
    start_epoch = 0
    num_workers = 8 # 增加数据加载并行度，减少数据加载瓶颈
    print_freq = 5
    gpus = '0' # 使用第一个GPU进行训练
    checkpoint = None
    checkpoint_dir = './checkpoint_mdf_rgb'
