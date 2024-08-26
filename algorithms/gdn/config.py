class Conf:
    batch_size = 512
    epoch = 100
    window_size = 30
    slide_stride = 1
    dim = 64
    topk = 10  # 몇개의 top-k를 사용할건지
    out_layer_inter_dim = 256
    out_layer_num = 1
    val_ratio = 0.2
    decay = 0
    report = 'best'
    threshold = 99

    env_config = {
        'report': report,
        'device': 'cpu',
        'save_path': None
    }