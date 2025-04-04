# ComfyUI Super Resolution

A collection of high-performance neural network-based Super Resolution models for ComfyUI.

## Features

- **Multiple SR Models**: Choose from FSRCNN, ESPCN, LapSRN, and EDSR with different quality/speed tradeoffs
- **Modular Design**: Load models once and reuse them across multiple upscaling operations
- **CUDA Acceleration**: Automatic GPU acceleration when available
- **Multiple Scale Factors**: Support for 2x, 3x, and 4x upscaling

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI_SuperResolution
   ```

2. Restart ComfyUI - the necessary models will be automatically downloaded on first use

## Usage

1. Add a **SR Model Loader** node to your workflow
2. Select your desired model type and scale factor
3. Connect the model output to a **SR Upscale** node
4. Connect an image to the upscale node
5. Run your workflow to get high-quality upscaled images!

## Model Comparison

| Model | Architecture | Features | Best For | Speed | Quality |
|-------|-------------|----------|----------|-------|---------|
| **FSRCNN-small** | Lightweight CNN | Fast, minimal memory use | Real-time processing, mobile | ★★★★★ | ★★ |
| **FSRCNN** | CNN with larger features | Good balance of speed/quality | General purpose | ★★★★ | ★★★ |
| **ESPCN** | Sub-pixel convolutions | Efficient upscaling at end | Text/line drawings | ★★★★ | ★★★ |
| **VDSR** | Very deep CNN | Better edge reconstruction | Detailed images with edges | ★★★ | ★★★★ |
| **LapSRN** | Laplacian pyramid | Progressive upscaling | Sharp edges, details | ★★★ | ★★★★ |
| **EDSR** | Deep residual network | Most parameters, best quality | Maximum detail | ★★ | ★★★★★ |

## Technical Details

Each model has unique architectural characteristics:

1. **FSRCNN / FSRCNN-small**: Direct mapping from low to high resolution, lightweight with fewer parameters.

2. **ESPCN**: Uses the efficient sub-pixel convolution technique that processes at low resolution and only expands dimensions at the final layer.

3. **VDSR**: Very Deep Super Resolution network with 20 convolutional layers, uses global residual learning.

4. **LapSRN**: Uses a Laplacian pyramid structure to progressively upscale images, which preserves edges better than the others.

5. **EDSR**: Enhanced Deep Super-Resolution, a residual network with significantly more parameters, offering the highest quality but slower processing.

## Performance Expectations (RTX 4090)

| Model | 2x Upscaling | 4x Upscaling | Memory Usage |
|-------|--------------|--------------|--------------|
| FSRCNN-small | 200+ FPS | 150+ FPS | < 100MB |
| FSRCNN | 100+ FPS | 80+ FPS | < 150MB |
| ESPCN | 90+ FPS | 70+ FPS | < 150MB |
| LapSRN | 60+ FPS | 40+ FPS | < 200MB |
| EDSR | 40+ FPS | 25+ FPS | < 500MB |

## When to Use Each Model

- **FSRCNN-small**: When maximum speed is required
- **FSRCNN**: For a good balance of quality and performance
- **ESPCN**: For text and line art upscaling
- **LapSRN**: When you need better edge preservation than FSRCNN
- **EDSR**: When maximum quality is desired and performance is secondary

## Comparison to Traditional Methods

These neural network-based upscaling methods offer significantly better quality than traditional algorithms like Lanczos, Bicubic, or Bilinear interpolation, while often being just as fast thanks to CUDA acceleration.

## Credits

This nodepack implements models originally created by:

- FSRCNN: Dong et al. (https://github.com/ryanontheinside/FSRCNN_Tensorflow)
- EDSR: Lim et al. (https://github.com/ryanontheinside/EDSR_Tensorflow)
- ESPCN: Shi et al. (https://github.com/ryanontheinside/TF-ESPCN)
- LapSRN: Lai et al. (https://github.com/ryanontheinside/TF-LapSRN)

## License

MIT 