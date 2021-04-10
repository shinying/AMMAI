from argparse import Namespace


def get_config(dataset):
    config = Namespace()
    config.dataset = dataset
    config.embedding_size = 512
    config.sample_rate = 1
    config.fp16 = False
    config.momentum = 0.9
    config.weight_decay = 5e-4
    config.batch_size = 128
    config.test_batch_size = 256
    config.lr = 0.1
    config.dropout = 0.5
    config.output = f"ms1mv3_arcface_r50_lr{config.lr}_b{config.batch_size}_d{config.dropout}"
    config.num_workers = 8

    # Train images preprocessed by MTCNN
    if config.dataset == 'APD':
        config.rec = "/tmp3/shinlee/ammai/hw1/APD1/train"
        config.output = "ckpt/" + config.output
        config.num_classes = 679
        config.num_image = 12899
        config.num_epoch = 40
        config.warmup_epoch = 1
        config.val_split = 0.2

        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
                [m for m in [20, 30, 38] if m - 1 <= epoch])
        config.lr_func = lr_step_func
        
    # Train images preprocessed by pretrained RetinaFace
    elif config.dataset == 'APD2':  
        config.rec = "/tmp3/shinlee/ammai/hw1/APD2/train"
        config.output = "ckpt2/" + config.output
        config.num_classes = 676
        config.num_image = 12905
        config.num_epoch = 45
        config.warmup_epoch = 1
        config.val_split = 0.2

        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
                [m for m in [30, 38, 43] if m - 1 <= epoch])
        config.lr_func = lr_step_func

    # Train images preprocessed by our RetinaFace
    elif config.dataset == "APD3":
        config.rec = "/tmp3/shinlee/ammai/hw1/APD3/train"
        config.output = "ckpt3/" + config.output
        config.num_classes = 678
        config.num_image = 13163
        config.num_epoch = 45
        config.warmup_epoch = 1
        config.val_split = 0.2

        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
                [m for m in [30, 38, 43] if m - 1 <= epoch])
        config.lr_func = lr_step_func
        
    else:
        raise NotImplementedError(f'Unsupported dataset: {config.dataset}')

    return config