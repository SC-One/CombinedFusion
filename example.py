
import torch
from CombinedFusion.CombinedFusion import CombinedFusion
from utils.utils import visualize_image_and_depth, load_imageURL_as_tensor

def main():
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        raise "Force warn: using GPU!"

    cfModel = CombinedFusion()
    cfModel.load_state_dict(torch.load('./CombinedFusion.pth', map_location='cpu'))
    cfModel = cfModel.to(DEVICE).eval()
    
    # replace your code:
    import torch.nn.functional as F
    img_tensor = load_imageURL_as_tensor("https://raw.githubusercontent.com/SC-One/CombinedFusion/refs/heads/master/assets/Tunnel.jpg")
    depth_tensor = cfModel(img_tensor.to(DEVICE).unsqueeze(0), max_depth=80)
    pred = F.interpolate(depth_tensor[:, None], img_tensor.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
    visualize_image_and_depth(img_tensor, pred)
    
if __name__ == '__main__':
    main()