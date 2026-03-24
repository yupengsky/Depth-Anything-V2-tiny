import argparse
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PATCH_SIZE = 14
INPUT_SIZE = 518
MAX_DEPTH = 20.0
BACKBONE_EMBED_DIM = 1024
BACKBONE_DEPTH = 24
BACKBONE_HEADS = 16
INTERMEDIATE_LAYERS = (4, 11, 17, 23)

DPT_FEATURES = 256
DPT_OUT_CHANNELS = (256, 512, 1024, 1024)
PREVIEW_ROWS = 6
PREVIEW_COLS = 6

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = BASE_DIR / "photos" / "1.jpg"
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "depth_anything_v2_metric_hypersim_vitl.pth"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def constrain_to_multiple(value, multiple, min_value):
    rounded = int(np.round(value / multiple) * multiple)
    if rounded < min_value:
        rounded = int(np.ceil(value / multiple) * multiple)
    return rounded


def resize_for_encoder(image, input_size, upscale_small_images=True):
    height, width = image.shape[:2]
    scale = max(input_size / height, input_size / width)

    min_target = input_size
    if not upscale_small_images:
        scale = min(scale, 1.0)
        min_target = PATCH_SIZE

    resized_height = constrain_to_multiple(scale * height, PATCH_SIZE, min_target)
    resized_width = constrain_to_multiple(scale * width, PATCH_SIZE, min_target)

    if resized_height == height and resized_width == width:
        return image

    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)


def prepare_image_tensor(raw_image, input_size, device, upscale_small_images=True):
    resized_image = resize_for_encoder(raw_image, input_size, upscale_small_images=upscale_small_images)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized_image = (resized_image - IMAGE_MEAN) / IMAGE_STD
    resized_image = np.transpose(resized_image, (2, 0, 1)).copy()
    tensor = torch.from_numpy(resized_image).unsqueeze(0).to(device)
    return tensor


def compress_image(raw_image, max_side):
    height, width = raw_image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return raw_image, 1.0

    scale = max_side / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    compressed = cv2.resize(raw_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return compressed, scale


def read_image(image_path):
    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return raw_image


def depth_to_vis(depth):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max == depth_min:
        return np.zeros_like(depth, dtype=np.uint8)
    return ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)


def preview_depth_matrix(depth, rows=PREVIEW_ROWS, cols=PREVIEW_COLS):
    row_indices = np.linspace(0, depth.shape[0] - 1, num=min(rows, depth.shape[0]), dtype=int)
    col_indices = np.linspace(0, depth.shape[1] - 1, num=min(cols, depth.shape[1]), dtype=int)
    return depth[np.ix_(row_indices, col_indices)]


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = (INPUT_SIZE // PATCH_SIZE) ** 2
        self.proj = nn.Conv2d(3, BACKBONE_EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = BACKBONE_HEADS
        self.scale = (BACKBONE_EMBED_DIM // BACKBONE_HEADS) ** -0.5
        self.qkv = nn.Linear(BACKBONE_EMBED_DIM, BACKBONE_EMBED_DIM * 3, bias=True)
        self.proj = nn.Linear(BACKBONE_EMBED_DIM, BACKBONE_EMBED_DIM, bias=True)

    def forward(self, x):
        batch_size, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, token_count, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attention = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attention @ v).transpose(1, 2).reshape(batch_size, token_count, channels)
        return self.proj(x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = BACKBONE_EMBED_DIM * 4
        self.fc1 = nn.Linear(BACKBONE_EMBED_DIM, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, BACKBONE_EMBED_DIM, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(BACKBONE_EMBED_DIM, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale(BACKBONE_EMBED_DIM, init_value=1.0)
        self.norm2 = nn.LayerNorm(BACKBONE_EMBED_DIM, eps=1e-6)
        self.mlp = Mlp()
        self.ls2 = LayerScale(BACKBONE_EMBED_DIM, init_value=1.0)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class ViTLargeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, BACKBONE_EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, BACKBONE_EMBED_DIM))
        self.mask_token = nn.Parameter(torch.zeros(1, BACKBONE_EMBED_DIM))
        self.blocks = nn.ModuleList([Block() for _ in range(BACKBONE_DEPTH)])
        self.norm = nn.LayerNorm(BACKBONE_EMBED_DIM, eps=1e-6)

    def interpolate_pos_encoding(self, x, height, width):
        patch_tokens = x.shape[1] - 1
        base_patch_tokens = self.pos_embed.shape[1] - 1
        target_height = height // self.patch_size
        target_width = width // self.patch_size

        if patch_tokens == base_patch_tokens and target_height == target_width:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]
        embed_dim = x.shape[-1]
        base_size = int(math.sqrt(base_patch_tokens))
        target_height_with_offset = target_height + 0.1
        target_width_with_offset = target_width + 0.1

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, base_size, base_size, embed_dim).permute(0, 3, 1, 2).float(),
            scale_factor=(target_height_with_offset / base_size, target_width_with_offset / base_size),
            mode="bicubic",
            antialias=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(dtype=x.dtype)

    def prepare_tokens(self, x):
        batch_size, _, height, width = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, height, width)
        return x

    def get_intermediate_layers(self, x):
        x = self.prepare_tokens(x)
        outputs = []
        target_layers = set(INTERMEDIATE_LAYERS)

        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in target_layers:
                outputs.append(self.norm(x))

        return tuple((out[:, 1:], out[:, 0]) for out in outputs)


def make_scratch(in_channels, out_channels):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_channels[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_channels[1], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_channels[2], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_channels[3], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(x, inplace=False)
        out = self.conv1(out)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, x, residual=None, size=None):
        if residual is not None:
            x = x + self.resConfUnit1(residual)

        x = self.resConfUnit2(x)
        resize_kwargs = {"mode": "bilinear", "align_corners": True}
        if size is None:
            x = F.interpolate(x, scale_factor=2, **resize_kwargs)
        else:
            x = F.interpolate(x, size=size, **resize_kwargs)
        return self.out_conv(x)


class DPTHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(BACKBONE_EMBED_DIM, DPT_OUT_CHANNELS[0], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(BACKBONE_EMBED_DIM, DPT_OUT_CHANNELS[1], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(BACKBONE_EMBED_DIM, DPT_OUT_CHANNELS[2], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(BACKBONE_EMBED_DIM, DPT_OUT_CHANNELS[3], kernel_size=1, stride=1, padding=0),
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(DPT_OUT_CHANNELS[0], DPT_OUT_CHANNELS[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(DPT_OUT_CHANNELS[1], DPT_OUT_CHANNELS[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(DPT_OUT_CHANNELS[3], DPT_OUT_CHANNELS[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        self.scratch = make_scratch(DPT_OUT_CHANNELS, DPT_FEATURES)
        self.scratch.refinenet1 = FeatureFusionBlock(DPT_FEATURES)
        self.scratch.refinenet2 = FeatureFusionBlock(DPT_FEATURES)
        self.scratch.refinenet3 = FeatureFusionBlock(DPT_FEATURES)
        self.scratch.refinenet4 = FeatureFusionBlock(DPT_FEATURES)
        self.scratch.output_conv1 = nn.Conv2d(DPT_FEATURES, DPT_FEATURES // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(DPT_FEATURES // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, features, patch_height, patch_width):
        projected = []

        for feature, project, resize_layer in zip(features, self.projects, self.resize_layers):
            patch_tokens = feature[0]
            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(
                patch_tokens.shape[0],
                patch_tokens.shape[-1],
                patch_height,
                patch_width,
            )
            projected.append(resize_layer(project(patch_tokens)))

        layer_1, layer_2, layer_3, layer_4 = projected
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, residual=layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, residual=layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, residual=layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, size=(patch_height * PATCH_SIZE, patch_width * PATCH_SIZE), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out


class MetricDepthModel(nn.Module):
    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.pretrained = ViTLargeBackbone()
        self.depth_head = DPTHead()

    def forward(self, x):
        patch_height = x.shape[-2] // PATCH_SIZE
        patch_width = x.shape[-1] // PATCH_SIZE
        features = self.pretrained.get_intermediate_layers(x)
        depth = self.depth_head(features, patch_height, patch_width) * self.max_depth
        return depth.squeeze(1)

    @torch.inference_mode()
    def infer_image(self, raw_image, input_size, device, upscale_small_images=True):
        height, width = raw_image.shape[:2]
        image = prepare_image_tensor(raw_image, input_size, device, upscale_small_images=upscale_small_images)
        depth = self.forward(image)
        depth = F.interpolate(depth[:, None], size=(height, width), mode="bilinear", align_corners=True)[0, 0]
        return depth.cpu().numpy()


def build_model(checkpoint_path, max_depth, device):
    model = MetricDepthModel(max_depth=max_depth)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@lru_cache(maxsize=None)
def _build_model_cached(checkpoint_path_str, max_depth, device):
    return build_model(Path(checkpoint_path_str), max_depth, device)


def infer_depth_from_image(
    raw_image,
    checkpoint_path,
    input_size=INPUT_SIZE,
    max_depth=MAX_DEPTH,
    device=None,
    upscale_small_images=True,
):
    if device is None:
        device = get_device()

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = _build_model_cached(str(checkpoint_path), float(max_depth), str(device))
    depth = model.infer_image(raw_image, input_size, device, upscale_small_images=upscale_small_images)
    return depth, device


def infer_depth_from_path(
    image_path,
    checkpoint_path,
    input_size=INPUT_SIZE,
    max_depth=MAX_DEPTH,
    compress_max_side=1024,
    upscale_small_images=True,
):
    image_path = Path(image_path).resolve()
    raw_image = read_image(image_path)
    compressed_image, resize_scale = compress_image(raw_image, compress_max_side)
    depth, device = infer_depth_from_image(
        compressed_image,
        checkpoint_path,
        input_size=input_size,
        max_depth=max_depth,
        upscale_small_images=upscale_small_images,
    )
    return {
        "device": device,
        "raw_image": raw_image,
        "compressed_image": compressed_image,
        "resize_scale": resize_scale,
        "depth": depth,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Infer metric depth for a single photo.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Metric depth checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output depth files.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE, help="Inference input size.")
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH, help="Maximum metric depth in meters.")
    parser.add_argument("--compress-max-side", type=int, default=1024, help="Resize the input image once before inference.")
    parser.add_argument(
        "--no-upscale-small-image",
        action="store_true",
        help="Keep small compressed images near their current size instead of enlarging them back toward the encoder input size.",
    )
    parser.add_argument("--no-save", action="store_true", help="Print inference information without writing output files.")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    inference = infer_depth_from_path(
        image_path,
        checkpoint_path,
        input_size=args.input_size,
        max_depth=args.max_depth,
        compress_max_side=args.compress_max_side,
        upscale_small_images=not args.no_upscale_small_image,
    )
    raw_image = inference["raw_image"]
    compressed_image = inference["compressed_image"]
    resize_scale = inference["resize_scale"]
    depth = inference["depth"]
    device = inference["device"]

    if not args.no_save:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_output_path = output_dir / f"{image_path.stem}_depth.png"
        raw_output_path = output_dir / f"{image_path.stem}_depth.npy"
        cv2.imwrite(str(vis_output_path), depth_to_vis(depth))
        np.save(str(raw_output_path), depth)

    print(f"Device: {device}")
    print(f"Input image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Original image shape: {raw_image.shape[:2]}")
    print(f"Compressed image shape: {compressed_image.shape[:2]}")
    print(f"Resize scale: {resize_scale:.4f}")
    print(f"Upscale small image before encoder: {not args.no_upscale_small_image}")
    if not args.no_save:
        print(f"Depth image saved to: {vis_output_path}")
        print(f"Raw depth matrix saved to: {raw_output_path}")
    print(f"Depth shape: {depth.shape}")
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print("Depth matrix preview (meters):")
    print(preview_depth_matrix(depth))


if __name__ == "__main__":
    main()
