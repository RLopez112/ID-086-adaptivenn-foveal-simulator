import torch

def main():
    checkpoint_path = 'AdaptiveNN_ckpt.pth'
    print(f"Loading checkpoint: {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['model'] if 'model' in ckpt else ckpt

    top_modules = {}
    for k, v in sd.items():
        prefix = k.split('.')[0]
        if prefix not in top_modules:
            top_modules[prefix] = {'tensors': 0, 'params': 0}
        top_modules[prefix]['tensors'] += 1
        top_modules[prefix]['params'] += v.numel()

    fmt = "%-40s | %7s | %16s | %8s"
    print("\n" + "=" * 80)
    print(fmt % ("Model Attribute (dynamic_deitS.py)", "Tensors", "Exact Parameters", "Approx"))
    print("-" * 80)
    
    total_params = 0
    for mod, info in sorted(top_modules.items()):
        total_params += info['params']
        p_count = info['params']
        approx_m = p_count / 1e6
        print(fmt % (mod, info['tensors'], f"{p_count:,}", f"~{approx_m:.1f}M"))

    print("=" * 80)
    print(fmt % ("GRAND TOTAL", len(sd), f"{total_params:,}", f"~{total_params/1e6:.1f}M"))
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
