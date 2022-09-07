import argparse
import os
import paddle
import torch
from realesrgan.archs.rrdbnet_arch import  RRDBNet



device = paddle.get_device()
os.environ['CUDA_VISIBLE_DEVICES'] = device.replace('gpu:','')

def main(args):
    print(args)
    # An instance of the model
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    model.eval()

    torch_ckpt = torch.load(args.input, map_location=torch.device('cpu'))
    # print(torch_ckpt.keys())

    paddle_sd = {}
    for k, v in torch_ckpt['params_ema'].items():
        if torch.is_tensor(v):
            if 'conv_up1' in k or 'conv_up2' in k:
                # paddle_sd[k] = v.t().numpy() #如果是2维度
                print(v.shape)
                if 'weight' in k:
                    paddle_sd[k] = v.permute(0, 1, 3, 2).numpy()   #shape为[64, 64, 3, 3]
                else:
                    paddle_sd[k] = v.numpy()
            else:
                paddle_sd[k] = v.numpy()
        else:
            paddle_sd[k] = v
    paddle_ckpt = {'params_ema': paddle_sd}
    model.set_state_dict(paddle_ckpt)
    new_model_path= 'experiments/pretrained_models/RealESRGAN_x4plus.pdparams'
    paddle.save(paddle_ckpt, new_model_path )
    # check pdparams model
    model = paddle.load(new_model_path)

    print(model)
    print(model.keys())
    print(model['params_ema']['conv_first.weight'])


if __name__ == '__main__':
    """Convert pytorch model to paddle models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='experiments/pretrained_models/RealESRGAN_x4plus.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='realesrgan-x4.pdparams', help='Output paddle models path')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)




#
#
#
# torch_ckpt = torch.load('./experiments/pretrained_models/RealESRGAN_x4plus.pth', map_location=torch.device('cpu'))
# m = torch_ckpt['model']
# sd = m['sd']
# paddle_sd = {}
# for k, v in sd.items():
#
#     if torch.is_tensor(v):
#         if 'imnet.layers' in k and 'weight' in k:
#             print(k)
#             print(v)
#             paddle_sd[k] = v.t().numpy()
#         else:
#             paddle_sd[k] = v.numpy()
#     else:
#         paddle_sd[k] = v
#
# paddle_ckpt = {'name': m['name'], 'args': m['args'], 'sd': paddle_sd}
#
# net.set_state_dict(paddle_ckpt)
# paddle.save({'model': paddle_ckpt}, './pretrained/rdn-liif_torch.pdparams')