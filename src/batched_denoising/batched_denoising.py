import torch
import comfy.model_management
import comfy.samplers
import math

class StreamBatchSampler:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg_type": (["none", "full", "self", "initialize"], {"default": "self", "tooltip": "'self' (RCFG) is fastest, 'full' is standard SD but slower, 'initialize' is a middle ground, 'none' disables guidance"}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Only used with 'self' CFG. Controls strength of self-guidance. Higher = stronger guidance but more artifacts. Default 1.0", "round": 0.01}),
            },
        }
    
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "update"
    CATEGORY = "StreamPack/sampling"
    DESCRIPTION = "Implements batched denoising with different CFG options (including RCFG) for faster inference and temporal consistency."
    
    def __init__(self):
        self.frame_buffer = []
        self.x_t_latent_buffer = None
        self.stock_noise = None
        self.is_txt2img_mode = False
        self.delta = 1.0 
        self.cfg_type = "self"
        
        # Initialize all buffers
        self.zeros_reference = None
        self.random_noise_buffer = None
        self.sigmas_view_buffer = None
        self.expanded_stock_noise = None
        self.working_buffer = None
        self.output_buffer = None
        
        # Add RCFG state from reference
        self.cached_sigmas = None
        self.alpha_prod_t = None
        self.beta_prod_t = None
        self.alpha_prod_t_sqrt = None
        self.beta_prod_t_sqrt = None
        self.init_noise = None # Will be initialized in sample based on input shape
        self.step_counter = 0

    def compute_alpha_beta(self, sigmas):
        """Pre-compute alpha and beta terms for the given sigmas (LCM-like using cosine schedule)"""
        self.cached_sigmas = sigmas.clone()
        print(f"\n[Alpha/Beta] Input sigmas: {sigmas}")

        num_train_timesteps = 1000  # Standard for many diffusion models
        timesteps = torch.linspace(
            1, 0, num_train_timesteps, dtype=torch.float32, device=sigmas.device
        )
        alphas_cumprod = torch.cos((timesteps + 0.008) / 1.008 * math.pi / 2).pow(2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # Map sigmas to timesteps
        scaled_sigmas = (
            sigmas / sigmas.max() if sigmas.max() > 0 else torch.zeros_like(sigmas)
        )
        timesteps_for_sigma = (1 - scaled_sigmas) * (num_train_timesteps - 1)
        indices = (
            torch.round(timesteps_for_sigma).long().clamp(0, num_train_timesteps - 1)
        )

        alpha_prod_t = alphas_cumprod[indices].view(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        self.alpha_prod_t = alpha_prod_t
        self.beta_prod_t = beta_prod_t
        self.alpha_prod_t_sqrt = torch.sqrt(alpha_prod_t)
        self.beta_prod_t_sqrt = torch.sqrt(beta_prod_t)

        print(f"[Alpha/Beta] Final stats:")
        print(
            f"  alpha_prod_t: min={self.alpha_prod_t.min():.4f}, max={self.alpha_prod_t.max():.4f}, mean={self.alpha_prod_t.mean():.4f}"
        )
        print(
            f"  beta_prod_t: min={self.beta_prod_t.min():.4f}, max={self.beta_prod_t.max():.4f}, mean={self.beta_prod_t.mean():.4f}"
        )
        print(
            f"  alpha_sqrt: min={self.alpha_prod_t_sqrt.min():.4f}, max={self.alpha_prod_t_sqrt.max():.4f}, mean={self.alpha_prod_t_sqrt.mean():.4f}"
        )
        print(
            f"  beta_sqrt: min={self.beta_prod_t_sqrt.min():.4f}, max={self.beta_prod_t_sqrt.max():.4f}, mean={self.beta_prod_t_sqrt.mean():.4f}\n"
        )

    def scheduler_step_batch(self, denoised_batch, scaled_noise):
            """Compute scheduler step for the entire batch (LCM-like)"""
            print("\n[Scheduler] Starting scheduler step (LCM-like):")
            print(f"  Denoised shape: {denoised_batch.shape}")
            print(f"  Scaled noise shape: {scaled_noise.shape}")
            print(
                f"  Denoised stats: min={denoised_batch.min():.4f}, max={denoised_batch.max():.4f}, mean={denoised_batch.mean():.4f}"
            )
            print(
                f"  Scaled noise stats: min={scaled_noise.min():.4f}, max={scaled_noise.max():.4f}, mean={scaled_noise.mean():.4f}"
            )

            # Assuming self.alpha_prod_t and self.beta_prod_t are for the current timestep
            alpha_t_sqrt = self.alpha_prod_t_sqrt[: denoised_batch.shape[0]]
            beta_t_sqrt = self.beta_prod_t_sqrt[: denoised_batch.shape[0]]

            # Calculate the predicted original sample (x_0) from the noisy sample and model output
            # This is similar to the inverse of the forward diffusion step
            pred_original_sample = (
                denoised_batch - beta_t_sqrt.view(-1, 1, 1, 1) * scaled_noise
            ) / (alpha_t_sqrt.view(-1, 1, 1, 1) + 1e-7)
            print(
                f"  Predicted original sample stats: min={pred_original_sample.min():.4f}, max={pred_original_sample.max():.4f}, mean={pred_original_sample.mean():.4f}"
            )

            # For the reverse step, we need the alpha and beta for the previous timestep
            # Since we are operating on a batch of timesteps, we'll use the values corresponding to the first timestep in the batch for updating stock_noise
            # In StreamDiffusion paper, the update happens based on the result of the last denoising step

            # Get alpha and beta for the next step: (which would be the previous in the denoising sequence)
            alpha_next_sqrt = torch.cat(
                [
                    self.alpha_prod_t_sqrt[1 : denoised_batch.shape[0]],
                    torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                ],
                dim=0,
            )

            # Calculate the coefficient for the predicted original sample in the next step
            sigma_next = (1 - alpha_next_sqrt.pow(2)) / (alpha_next_sqrt + 1e-7)
            sigma_t = (1 - alpha_t_sqrt.pow(2)) / (alpha_t_sqrt + 1e-7)

            # Guidance scale (set to 1 for unconditional prediction)
            guidance_scale = 1

            # Calculate delta_x (the change in the noisy sample)
            delta_x = (sigma_next - sigma_t) * guidance_scale * pred_original_sample

            print(
                f"  Final delta_x stats: min={delta_x.min():.4f}, max={delta_x.max():.4f}, mean={delta_x.mean():.4f}\n"
            )

            return delta_x

    def sample(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        """Sample with staggered batch denoising steps - Optimized version with multiple CFG types"""
        extra_args = {} if extra_args is None else extra_args
        cond_scale = extra_args.get("model_options", {}).get("cfg_scale", 1.0) # Get cond_scale from model options
        cfg_type = self.cfg_type
        
        # Get number of frames in batch and available sigmas
        batch_size = noise.shape[0]
        num_sigmas = len(sigmas) - 1  # Subtract 1 because last sigma is the target (0.0)
        
        # Precompute alpha/beta if needed
        self.compute_alpha_beta(sigmas) # Sigmas include the final 0.0, compute uses all
        
        # Reuse zeros buffer for txt2img detection
        if self.zeros_reference is None or self.zeros_reference.device != noise.device or self.zeros_reference.dtype != noise.dtype:
             self.zeros_reference = torch.zeros(1, device=noise.device, dtype=noise.dtype)
        
        # Check if noise tensor is all zeros
        self.is_txt2img_mode = torch.abs(noise).sum() < 1e-5
        
        # Initialize init_noise sequence like reference if needed (once or if shape changes)
        noise_shape_base = noise.shape[1:] # C, H, W
        if self.init_noise is None or self.init_noise.shape[0] != num_sigmas or self.init_noise.shape[1:] != noise_shape_base:
            self.init_noise = torch.randn(
                (num_sigmas, *noise_shape_base),
                device=noise.device,
                dtype=noise.dtype,
            )
            # Initialize stock_noise to zeros when init_noise is created (like reference)
            self.stock_noise = torch.zeros_like(self.init_noise[0]) 
            self.step_counter = 0 # Reset step counter when noise shape changes

        # Shift noise sequence
        if self.step_counter > 0:
            self.init_noise = torch.cat([self.init_noise[1:], self.init_noise[0:1]], dim=0)
            # Update stock noise based on the *first* element of the *shifted* init_noise
            self.stock_noise = self.init_noise[0].clone() 


        if self.is_txt2img_mode:
            # If txt2img mode, use the init_noise sequence directly as the base noise for the batch
            # This ensures consistent noise across steps/frames for txt2img
            if self.random_noise_buffer is None or self.random_noise_buffer.shape != noise.shape:
                self.random_noise_buffer = torch.empty_like(noise)
            
            # Copy from the init_noise sequence for the current batch size
            self.random_noise_buffer.copy_(self.init_noise[:batch_size])
            x = self.random_noise_buffer  # Use pre-allocated buffer
        else:
             # If not txt2img, we'll still need to add noise later based on sigmas
            x = noise # Start with the input noise (img2img)
        
        # Verify batch size matches number of timesteps
        if batch_size != num_sigmas:
            raise ValueError(f"Batch size ({batch_size}) must match number of sigmas ({num_sigmas})")
        
        # Pre-allocate and reuse view buffer for sigmas
        # Ensure buffer matches the number of steps *in this batch* (num_sigmas)
        if self.sigmas_view_buffer is None or self.sigmas_view_buffer.shape[0] != num_sigmas:
            self.sigmas_view_buffer = torch.empty((num_sigmas, 1, 1, 1), 
                                               device=sigmas.device, 
                                               dtype=sigmas.dtype)
        # In-place copy of sigmas view (use sigmas[:-1] as these are the steps)
        self.sigmas_view_buffer.copy_(sigmas[:-1].view(-1, 1, 1, 1))
        
        # Apply noise with pre-allocated buffers - no new memory allocation
        if not self.is_txt2img_mode: 
            # For img2img, add noise based on sigmas and init_noise sequence
            # Expand the relevant part of init_noise for the batch
            current_batch_init_noise = self.init_noise[:batch_size].to(x.device) # Ensure device match
            
            # If x is the same object as noise, use a working buffer
            if id(x) == id(noise):  
                if self.working_buffer is None or self.working_buffer.shape != noise.shape:
                    self.working_buffer = torch.empty_like(noise)
                x = self.working_buffer
                torch.add(noise, current_batch_init_noise * self.sigmas_view_buffer, out=x)
            else: # Otherwise, add noise directly to x
                torch.add(noise, current_batch_init_noise * self.sigmas_view_buffer, out=x)

        # Initialize and manage latent buffer with memory optimization
        # Only buffer if more than one step
        if num_sigmas > 1:
            if self.x_t_latent_buffer is None or self.x_t_latent_buffer.shape != x[0].shape or self.is_txt2img_mode:
                # Pre-allocate or resize as needed
                if self.x_t_latent_buffer is None or self.x_t_latent_buffer.shape != x[0].shape:
                    self.x_t_latent_buffer = torch.empty_like(x[0])
                self.x_t_latent_buffer.copy_(x[0])
            
            # Use buffer for first frame to maintain temporal consistency
            x[0].copy_(self.x_t_latent_buffer)
        else: # Single step, no buffering needed
             self.x_t_latent_buffer = None 
        
        with torch.no_grad():
            # Process all frames in parallel
            sigma_batch = sigmas[:-1] # Use sigmas for the steps, excluding the final 0.0
            
            # --- CFG Handling --- 
            denoised_batch = None
            stock_noise_update_value = None # Track value to update stock noise with (used in self-cfg)
            
            # 1. No CFG
            if cfg_type == "none" or cond_scale <= 1.0:
                denoised_batch = model(x, sigma_batch, **extra_args)
                stock_noise_update_value = denoised_batch[0].clone().detach()

            # 2. Full CFG
            elif cfg_type == "full":
                x_double = torch.cat([x, x], dim=0)
                sigma_double = torch.cat([sigma_batch, sigma_batch], dim=0)
                model_output = model(x_double, sigma_double, **extra_args)
                noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                denoised_batch = noise_pred_uncond + cond_scale * (noise_pred_text - noise_pred_uncond)
                # Update stock noise with uncond prediction (using first frame)
                stock_noise_update_value = noise_pred_uncond[0].clone().detach()

            # 3. Initialize CFG
            elif cfg_type == "initialize":
                x_plus_uc = torch.cat([x[0:1], x], dim=0) # Add first frame again for uncond
                sigma_plus_uc = torch.cat([sigma_batch[0:1], sigma_batch], dim=0)
                model_output = model(x_plus_uc, sigma_plus_uc, **extra_args)
                noise_pred_uncond = model_output[0:1] # Uncond is the first item
                noise_pred_cond = model_output[1:]     # Cond are the rest
                # Calculate final denoised batch using CFG scale
                denoised_batch = noise_pred_uncond + cond_scale * (noise_pred_cond - noise_pred_uncond) 
                # Update stock noise with uncond prediction 
                stock_noise_update_value = noise_pred_uncond[0].clone().detach()
            
            # 4. Self CFG (RCFG)
            elif cfg_type == "self":
                eps_c = model(x, sigma_batch, **extra_args) # Conditional prediction
                
                # Initialize stock noise if needed (first run after init_noise creation)
                if self.stock_noise is None: # Should have been initialized with init_noise
                     # Fallback: initialize with first frame prediction if somehow missed
                     self.stock_noise = eps_c[0].clone().detach()
                
                # RCFG equation
                # Use current stock_noise (which should be based on init_noise[0])
                noise_pred_uncond = self.stock_noise.to(eps_c.device) * self.delta
                denoised_batch = noise_pred_uncond + cond_scale * (eps_c - noise_pred_uncond)
                
                # Defer stock noise update until after scheduler step below
                # Store the conditional prediction for the scheduler step
                stock_noise_update_value = eps_c.clone().detach() # Need the whole batch for scheduler


            # --- CFG Handling End --- 

            # --- RCFG Stock Noise Update ---
            if cfg_type == "self" and stock_noise_update_value is not None:
                 # Get the noise sequence for the current batch
                 current_batch_init_noise = self.init_noise[:batch_size].to(denoised_batch.device)
                 
             
                 beta_sqrt_batch = self.beta_prod_t_sqrt[:batch_size].to(denoised_batch.device)
                 scaled_noise = beta_sqrt_batch * current_batch_init_noise # Use the init_noise for this step

                 delta_x = self.scheduler_step_batch(denoised_batch, scaled_noise)
                 num_steps_in_batch = denoised_batch.shape[0]
                 if self.alpha_prod_t_sqrt.shape[0] >= num_steps_in_batch:
                     one_tensor = torch.ones_like(self.alpha_prod_t_sqrt[0:1])
                     zero_tensor = torch.zeros_like(self.beta_prod_t_sqrt[0:1])
                     alpha_next_sqrt = torch.cat([self.alpha_prod_t_sqrt[1:num_steps_in_batch], one_tensor], dim=0).to(delta_x.device)
                     beta_next_sqrt = torch.cat([self.beta_prod_t_sqrt[1:num_steps_in_batch], zero_tensor], dim=0).to(delta_x.device)
                 else:
                     raise ValueError("Precomputed alpha/beta shapes mismatch for final scaling.")
                 
                 # Apply the scaling
                 scaled_delta_x = (alpha_next_sqrt / (beta_next_sqrt + 1e-7)) * delta_x
                 
                 # And the *scaled* delta_x calculated for the *first frame* of the batch
                 self.stock_noise += scaled_delta_x[0].to(self.stock_noise.device) # Add scaled delta for first frame
            elif stock_noise_update_value is not None:
                 # For non-RCFG modes, update stock noise simply with the first frame's value
                 if self.stock_noise is None or self.stock_noise.shape != stock_noise_update_value.shape:
                     self.stock_noise = stock_noise_update_value.clone().detach()
                 else:
                     self.stock_noise.copy_(stock_noise_update_value.detach())


            # Update buffer with intermediate results
            if num_sigmas > 1:
                # Store result from first frame as buffer for next iteration
                self.x_t_latent_buffer.copy_(denoised_batch[0])  # In-place update
                
                # Pre-allocate output buffer
                output_shape = (1, *denoised_batch[-1].shape)
                if self.output_buffer is None or self.output_buffer.shape != output_shape:
                    self.output_buffer = torch.empty(output_shape, 
                                                  device=denoised_batch.device,
                                                  dtype=denoised_batch.dtype)
                # Copy the result directly to pre-allocated buffer
                self.output_buffer[0].copy_(denoised_batch[-1])
                x_0_pred_out = self.output_buffer
            else: # Single step
                x_0_pred_out = denoised_batch.unsqueeze(0) # Ensure batch dim [1,C,H,W]
                self.x_t_latent_buffer = None # No buffer needed
            
            # Call callback if provided
            if callback is not None:
                callback({'x': x_0_pred_out, 'i': self.step_counter, 'sigma': sigmas[0], 'sigma_hat': sigmas[0], 'denoised': denoised_batch[-1:]})
        
        # Increment step counter at the end of the sample method
        self.step_counter += 1
        
        return x_0_pred_out
    
    def update(self, cfg_type="self", delta=1.0):
        """Create sampler with specified settings"""
        self.cfg_type = cfg_type
        self.delta = delta 
        sampler = comfy.samplers.KSAMPLER(self.sample)
        return (sampler,)


class StreamScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "t_index_list": ("STRING", {
                    "default": "0,16,32,49",
                    "tooltip": "Comma-separated list of timesteps to actually use for denoising. Examples: '32,45' for img2img or '0,16,32,45' for txt2img"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Total number of timesteps in schedule. StreamDiffusion uses 50 by default. Only timesteps specified in t_index_list are actually used."
                }),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "update"
    CATEGORY = "StreamPack/sampling"
    DESCRIPTION = "Implements StreamDiffusion's efficient timestep selection. Use in conjunction with StreamBatchSampler."
    
    def update(self, model, t_index_list="32,45", num_inference_steps=50):
        # Get model's sampling parameters
        model_sampling = model.get_model_object("model_sampling")
        
        # Parse timestep list
        try:
            t_index_list = [int(t.strip()) for t in t_index_list.split(",")]
        except ValueError as e:
            
            t_index_list = [32, 45]
            
        # Create full schedule using normal scheduler
        full_sigmas = comfy.samplers.normal_scheduler(model_sampling, num_inference_steps)
        
        # Select only the sigmas at our desired indices, but in reverse order
        # This ensures we go from high noise to low noise
        selected_sigmas = []
        for t in sorted(t_index_list, reverse=True):  # Sort in reverse to go from high noise to low
            if t < 0 or t >= num_inference_steps:
                
                continue
            selected_sigmas.append(float(full_sigmas[t]))
            
        # Add final sigma
        selected_sigmas.append(0.0)
        
        # Convert to tensor and move to appropriate device
        selected_sigmas = torch.FloatTensor(selected_sigmas).to(comfy.model_management.get_torch_device())
        return (selected_sigmas,)


class StreamFrameBuffer:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "buffer_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of frames to buffer before starting batch processing. Should match number of denoising steps."
                }),
            },
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "update"
    CATEGORY = "StreamPack/sampling"
    DESCRIPTION = "Accumulates frames to enable staggered batch denoising like StreamDiffusion. Use in conjunction with StreamBatchSampler"
    
    
    def __init__(self):
        self.frame_buffer = None  # Tensor of shape [buffer_size, C, H, W]
        self.buffer_size = None
        self.buffer_pos = 0  # Current position
        self.is_initialized = False  # Track buffer initialization
        self.is_txt2img_mode = False
    
    def update(self, latent, buffer_size=4):
        """Add new frame to buffer and return batch when ready"""
        self.buffer_size = buffer_size
        
        # Extract latent tensor from input and remove batch dimension if present
        x = latent["samples"]
        
        # Check if this is a txt2img (zeros tensor) or img2img mode
        # In ComfyUI, EmptyLatentImage returns a zeros tensor with shape [batch_size, 4, h//8, w//8]
        # We consider it txt2img mode if the tensor contains all zeros
        is_txt2img_mode = torch.sum(torch.abs(x)) < 1e-6
        self.is_txt2img_mode = is_txt2img_mode
        
        # If it's a batch with size 1, remove the batch dimension for our buffer
        if x.dim() == 4 and x.shape[0] == 1:  # [1,C,H,W]
            x = x.squeeze(0)  # Remove batch dimension -> [C,H,W]
        
        # Initialize or resize frame_buffer as a tensor
        if not self.is_initialized or self.frame_buffer.shape[0] != self.buffer_size or \
           self.frame_buffer.shape[1:] != x.shape:
            # Pre-allocate buffer with correct shape
            self.frame_buffer = torch.zeros(
                (self.buffer_size, *x.shape),
                device=x.device,
                dtype=x.dtype
            )
            if self.is_txt2img_mode or not self.is_initialized:
                # Optimization: Use broadcasting to fill buffer with copies
                self.frame_buffer[:] = x.unsqueeze(0)  # Broadcast x to [buffer_size, C, H, W]

            self.is_initialized = True
            self.buffer_pos = 0
        else:
            # Add new frame to buffer using ring buffer logic
            self.frame_buffer[self.buffer_pos] = x  # In-place update
            self.buffer_pos = (self.buffer_pos + 1) % self.buffer_size  # Circular increment
        
        # Optimization: frame_buffer is already a tensor batch, no need to stack
        batch = self.frame_buffer

        
        # Return as latent dict with preserved dimensions
        result = {"samples": batch}
        
        # Preserve height and width if present in input
        if "height" in latent:
            result["height"] = latent["height"]
        if "width" in latent:
            result["width"] = latent["width"]
            
        return (result,)