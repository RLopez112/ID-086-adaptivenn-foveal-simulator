"""
AdaptiveNN Single Image Inference & Visualization Script
Runs dynamic glance-and-focus ViT inference using the trained PPO policy network.
"""

import os
import sys
import argparse

# Ensure local timm and models are in python path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'models'))

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from timm.models import create_model
import dynamic_deitS

try:
    from torchvision.models import ResNet50_Weights
    IMAGENET_CATEGORIES = ResNet50_Weights.DEFAULT.meta.get("categories", None)
except Exception:
    IMAGENET_CATEGORIES = None


def get_args_parser():
    parser = argparse.ArgumentParser('AdaptiveNN Single-Image Inference', add_help=True)
    parser.add_argument('--image', default='images/image_1.png', type=str,
                        help='Path to the input image')
    parser.add_argument('--checkpoint', default='AdaptiveNN_ckpt.pth', type=str,
                        help='Path to the pretrained checkpoint (.pth)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to run on ("cuda" or "cpu")')
    parser.add_argument('--seq_l', default=4, type=int,
                        help='Number of adaptive focus steps')
    parser.add_argument('--input_size', default=288, type=int,
                        help='Full input image resolution (default: 288)')
    parser.add_argument('--glance_input_size', default=112, type=int,
                        help='Glance network input resolution (default: 112)')
    parser.add_argument('--focus_patch_size', default=112, type=int,
                        help='Size of high-res patches cropped by policy (default: 112)')
    parser.add_argument('--focus_net_reg_size', default=160, type=int,
                        help='Focus network regularized input resolution')
    parser.add_argument('--topk', default=5, type=int,
                        help='Number of top prediction classes to show')
    parser.add_argument('--save_vis', default='outputs/vis_result.png', type=str,
                        help='File path to save the visualized policy crop trajectory (empty string to disable)')
    return parser


def get_category_name(class_id):
    if IMAGENET_CATEGORIES and 0 <= class_id < len(IMAGENET_CATEGORIES):
        return IMAGENET_CATEGORIES[class_id]
    return f"Class {class_id}"


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"[AdaptiveNN] Running on device: {device}")
    if device.type == 'cuda':
        print(f"[AdaptiveNN] GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}. Please download it first.")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found at: {args.image}")

    print(f"[AdaptiveNN] Initializing dynamic_deitS model...")
    model = create_model(
        'dynamic_deitS',
        seq_len=args.seq_l,
        feature_in_chans=384,
        policy_net_hidden_chans=128,
        policy_net_kernel_size=3,
        recover_n=5,
        remaining_blocks=4,
        glance_input_size=args.glance_input_size,
        glance_net_depth=12,
        glance_net_mlp_ratio=4,
        glance_net_drop_path=0.0,
        focus_patch_size=args.focus_patch_size,
        focus_net_reg_size=args.focus_net_reg_size,
        focus_net_depth=12,
        focus_net_mlp_ratio=4,
        focus_net_drop_path=0.0,
        multi_cls_drop_path=0.0,
        pretrained=False,
        num_classes=1000,
    )

    print(f"[AdaptiveNN] Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Load and preprocess image
    print(f"[AdaptiveNN] Loading input image: {args.image}")
    raw_img = Image.open(args.image).convert('RGB')
    crop_pct = 0.925
    resize_dim = int(args.input_size / crop_pct)

    eval_transform = transforms.Compose([
        transforms.Resize(resize_dim, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = eval_transform(raw_img).unsqueeze(0).to(device)

    print(f"[AdaptiveNN] Executing dynamic glance-to-focus inference (seq_l={args.seq_l})...")
    with torch.no_grad():
        outputs = model(input_tensor, seq_l=args.seq_l)

    # 1. Glance Stage Results
    glance_probs = F.softmax(outputs['x_glance'][0], dim=1)
    topk_glance = torch.topk(glance_probs, args.topk)

    print("\n" + "="*60)
    print("STAGE 1: GLANCE INFERENCE (Low-res 112x112 Coarse Pass)")
    print("="*60)
    for rank, (prob, idx) in enumerate(zip(topk_glance.values[0], topk_glance.indices[0]), start=1):
        c_name = get_category_name(idx.item())
        print(f"  #{rank}: {c_name:<30} (ID: {idx.item():<4}) | Confidence: {prob.item()*100:6.2f}%")

    # 2. Sequential Focus Policy Trajectory
    print("\n" + "="*60)
    print(f"STAGE 2: SEQUENTIAL FOCUS REFINEMENT ({args.seq_l} Steps)")
    print("="*60)
    actions = outputs['actions']
    focus_logits = outputs['x_focus']

    step_boxes = []
    for step_idx in range(args.seq_l):
        action = actions[step_idx][0].cpu().numpy()
        step_logits = focus_logits[step_idx]
        step_probs = F.softmax(step_logits, dim=1)
        best_class = torch.argmax(step_probs, dim=1).item()
        best_prob = step_probs[0, best_class].item()
        c_name = get_category_name(best_class)

        # Coordinate calculation matching dynamic_deitS patch extraction
        patch_size = args.focus_patch_size
        img_size = args.input_size
        y_center_norm, x_center_norm = action[0], action[1]
        
        y1 = int(action[0] * (img_size - patch_size))
        x1 = int(action[1] * (img_size - patch_size))
        y2 = y1 + patch_size
        x2 = x1 + patch_size
        step_boxes.append((x1, y1, x2, y2))

        print(f"  Step {step_idx + 1}: Policy Crop Center (y={y_center_norm:.3f}, x={x_center_norm:.3f})")
        print(f"          Box: [{x1}, {y1}] -> [{x2}, {y2}] ({patch_size}x{patch_size} px)")
        print(f"          Top: {c_name} | Confidence: {best_prob*100:.2f}%\n")

    # 3. Final Prediction
    final_probs = F.softmax(focus_logits[-1], dim=1)
    topk_final = torch.topk(final_probs, args.topk)

    print("="*60)
    print("FINAL CLASSIFICATION (Aggregated Focus Output)")
    print("="*60)
    for rank, (prob, idx) in enumerate(zip(topk_final.values[0], topk_final.indices[0]), start=1):
        c_name = get_category_name(idx.item())
        print(f"  #{rank}: {c_name:<30} (ID: {idx.item():<4}) | Confidence: {prob.item()*100:6.2f}%")
    print("="*60)

    # 4. Optional Visualization of Policy Bounding Boxes
    if args.save_vis:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_vis)), exist_ok=True)
        # Preprocess visual image identical to model input crop
        vis_base = transforms.Compose([
            transforms.Resize(resize_dim, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
        ])(raw_img)

        draw = ImageDraw.Draw(vis_base)
        colors = ['#FF3333', '#33FF33', '#3388FF', '#FFB733', '#FF33DA', '#00FFFF']

        for step_idx, (x1, y1, x2, y2) in enumerate(step_boxes):
            color = colors[step_idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"Step {step_idx+1}"
            draw.text((x1 + 4, y1 + 4), label, fill=color)

        vis_base.save(args.save_vis)
        print(f"\n[AdaptiveNN] Policy crop visualization saved to: {args.save_vis}")


if __name__ == '__main__':
    main()
