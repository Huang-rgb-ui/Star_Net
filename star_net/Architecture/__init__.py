import torch
import os

try:
    from .star_net_h import STARNetH
    from .star_net_l import STARNetL

    from .repvgg_unet import RepVGGUNet
    from .fasternet_unet import FasterNetUNet
    from .unet import UNet
    from .few_shot_panet import PANet
    from .swin_unet import SwinUNet
    from .pki_net import PKINet
    from .mambaout import MambaOut_UNet

except ImportError as e:
    print(f"⚠️ [Warning] Model import failed: {e}")


def model_generator(method, pretrained_model_path=None, **kwargs):
    method = method.lower().strip()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None

    star_args = {'in_channels': 110, 'tasks': ['grape', 'corn', 'tomato']}
    star_args.update(kwargs)
    base_args = {'in_channels': 110}
    base_args.update(kwargs)

    # === Ours ===
    if method == 'star_net_h':
        model = STARNetH(**star_args).to(device)
    elif method == 'star_net_l':
        model = STARNetL(**star_args).to(device)

    # === Baselines ===
    elif method == 'unet':
        model = UNet(**base_args).to(device)
    elif method == 'swin_unet':
        model = SwinUNet(**base_args).to(device)
    elif method == 'few_shot_panet':
        model = PANet(**base_args).to(device)
    elif method == 'repvgg_unet':
        model = RepVGGUNet(**base_args).to(device)
    elif method == 'fasternet_unet':
        model = FasterNetUNet(**base_args).to(device)
    elif method == 'pki_net':
        model = PKINet(**base_args).to(device)
    elif method == 'mambaout':
        model = MambaOut_UNet(**base_args).to(device)
    else:
        raise ValueError(f"❌ 未知的模型名称: {method}")

    if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        print(f'🔄 Loading from: {pretrained_model_path}')
        try:
            ckpt = torch.load(pretrained_model_path, map_location=device)
            sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            print('✅ Loaded')
        except Exception as e:
            print(f'❌ Failed: {e}')
    return model
