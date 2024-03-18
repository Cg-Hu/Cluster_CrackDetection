import argparse  
  
def get_args(parser=argparse.ArgumentParser()):
    # model
    parser.add_argument('--backbone_model', type=str, default='resnet18') # resnet18, resnet34, resnet50, swin_t, swin_s...
    parser.add_argument('--backbone_pretrained', type=int, default=0) #
    parser.add_argument('--weight_decay', type=float, default=0) #
    
    # optimizer
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # dataset
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--dataset_dir_path', nargs='+', type=str, default=['../data/crack_dataset_total'])
    # parser.add_argument('--dataset_dir_path', nargs='+', type=str, default=['../data/crack_dataset_test'])
    parser.add_argument('--enhanced', type=int, default=3) # 0, 1(spacial), 2(pixel), 3(spacial+pixel)
    
    # misc
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--seed', type=int, default=314)
    parser.add_argument('--desc', type=str, default='test', required=False)
    parser.add_argument('--output', action='store_true', default=True)  

    opt = parser.parse_args()  
    if opt.output:
        print(opt)
    return opt

if __name__ == '__main__':
    opt = get_args()