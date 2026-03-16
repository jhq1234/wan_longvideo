# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Long video generation via SyncTweedies:
# Overlapping temporal windows + predicted x0 weighted averaging.
# Reference: wan/video_synctweedies_protocol.md
import gc
import logging
import math
import random
import sys
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .text2video import WanT2V
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def _get_window_starts(T_total_lat: int, T_win_lat: int,
                       window_stride: int) -> list:
    """Return a list of latent-frame start indices for overlapping windows.

    Windows are placed so that:
    - The first window starts at 0.
    - The last window ends exactly at T_total_lat (clamped).
    - Intermediate windows are evenly spaced with the given stride.
    - All temporal positions are covered (coverage 100% guaranteed).
    """
    if T_total_lat <= T_win_lat:
        return [0]

    starts = []
    pos = 0
    while pos + T_win_lat < T_total_lat:
        starts.append(pos)
        pos += window_stride

    # Always include a window that covers the very end.
    last_start = T_total_lat - T_win_lat
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    return starts


def _get_window_starts_multiplier(window_size: int, multiplier: int,
                                  overlap_start: int, vae_stride: int) -> tuple:
    """Return window starts and total frames for multiplier-based chunking.

    Args:
        window_size: Number of pixel frames per window.
        multiplier: Number of chunks to generate.
        overlap_start: Pixel frame index where overlap starts within a chunk.
        vae_stride: VAE temporal stride.

    Returns:
        Tuple of (starts: list, T_total_lat: int, T_win_lat: int, overlap_start_lat: int)
        - starts: List of latent frame start positions for each chunk.
        - T_total_lat: Total number of latent frames.
        - T_win_lat: Number of latent frames per window.
        - overlap_start_lat: Latent frame index where overlap starts within a chunk.
    """
    if multiplier == 1:
        T_win_lat = (window_size - 1) // vae_stride + 1
        return [0], T_win_lat, T_win_lat, overlap_start // vae_stride

    # Calculate latent dimensions
    T_win_lat = (window_size - 1) // vae_stride + 1
    overlap_start_lat = overlap_start // vae_stride
    
    # Calculate stride in pixel frames and latent frames
    stride_pixel = window_size - overlap_start
    stride_lat = stride_pixel // vae_stride

    # Calculate total latent frames
    # First chunk: T_win_lat frames
    # Each additional chunk adds stride_lat new frames
    T_total_lat = T_win_lat + (multiplier - 1) * stride_lat

    # Calculate window starts: chunk i starts at i * stride_lat
    starts = []
    for i in range(multiplier):
        starts.append(i * stride_lat)

    return starts, T_total_lat, T_win_lat, overlap_start_lat


def _extract_windows(video_latent: torch.Tensor, starts: list,
                     T_win_lat: int) -> list:
    """Extract window tensors from the full video latent buffer.

    Args:
        video_latent: [C, T_total_lat, H, W]
        starts: list of integer latent frame start positions
        T_win_lat: number of latent frames per window

    Returns:
        List of [C, T_win_lat, H, W] tensors (copies).
    """
    return [
        video_latent[:, s:s + T_win_lat, :, :].clone() for s in starts
    ]


def _aggregate_x0(x0_preds: list, starts: list, T_win_lat: int,
                  T_total_lat: int, C: int, H: int, W: int,
                  device: torch.device, avg_mode: bool) -> torch.Tensor:
    """Aggregate per-window x0 predictions back into the full video latent.

    Args:
        x0_preds: list of [C, T_win_lat, H, W] tensors
        starts:   list of latent frame start positions (same length as x0_preds)
        avg_mode: if True, average overlapping predictions;
                  if False, last write wins (individual mode).

    Returns:
        Full video x̂₀ [C, T_total_lat, H, W].
    """
    if avg_mode:
        buf = torch.zeros(C, T_total_lat, H, W, device=device,
                          dtype=x0_preds[0].dtype)
        cnt = torch.zeros(T_total_lat, device=device, dtype=torch.float32)
        for x0, s in zip(x0_preds, starts):
            buf[:, s:s + T_win_lat, :, :] += x0
            cnt[s:s + T_win_lat] += 1.0
        # cnt is guaranteed >= 1 by coverage construction
        return buf / cnt[None, :, None, None].clamp(min=1.0)
    else:
        buf = torch.empty(C, T_total_lat, H, W, device=device,
                          dtype=x0_preds[0].dtype)
        for x0, s in zip(x0_preds, starts):
            buf[:, s:s + T_win_lat, :, :] = x0
        return buf


def _aggregate_x0_weighted(x0_preds: list, starts: list, T_win_lat: int,
                            T_total_lat: int, overlap_start_lat: int,
                            C: int, H: int, W: int,
                            device: torch.device) -> torch.Tensor:
    """Aggregate per-window x0 predictions with linear weighted averaging in overlap regions.

    Args:
        x0_preds: list of [C, T_win_lat, H, W] tensors
        starts:   list of latent frame start positions (same length as x0_preds)
        T_win_lat: number of latent frames per window
        T_total_lat: total number of latent frames
        overlap_start_lat: latent frame index where overlap starts within a chunk
        C, H, W: channel, height, width dimensions
        device: torch device

    Returns:
        Full video x̂₀ [C, T_total_lat, H, W] with weighted averaging in overlap regions.
    """
    buf = torch.zeros(C, T_total_lat, H, W, device=device,
                      dtype=x0_preds[0].dtype)
    weight_sum = torch.zeros(T_total_lat, device=device, dtype=torch.float32)

    num_chunks = len(x0_preds)
    overlap_length = T_win_lat - overlap_start_lat

    for chunk_idx, (x0, start) in enumerate(zip(x0_preds, starts)):
        chunk_end = start + T_win_lat
        
        # Non-overlapping region before overlap
        non_overlap_start = start
        non_overlap_end = start + overlap_start_lat
        
        # Check if this region overlaps with previous chunk
        if chunk_idx == 0:
            # First chunk: use all frames before overlap_start
            if non_overlap_start < non_overlap_end:
                buf[:, non_overlap_start:non_overlap_end, :, :] = \
                    x0[:, :non_overlap_end - non_overlap_start, :, :]
                weight_sum[non_overlap_start:non_overlap_end] = 1.0
        else:
            # Later chunks: only write if not already covered
            prev_chunk_end = starts[chunk_idx - 1] + T_win_lat
            if prev_chunk_end <= non_overlap_start and non_overlap_start < non_overlap_end:
                buf[:, non_overlap_start:non_overlap_end, :, :] = \
                    x0[:, :non_overlap_end - non_overlap_start, :, :]
                weight_sum[non_overlap_start:non_overlap_end] = 1.0

        # Overlapping region: linear weighted average with next chunk
        if chunk_idx < num_chunks - 1:
            next_chunk_start = starts[chunk_idx + 1]
            overlap_region_start = start + overlap_start_lat
            overlap_region_end = min(chunk_end, next_chunk_start + T_win_lat, T_total_lat)
            
            if overlap_region_start < overlap_region_end:
                overlap_size = overlap_region_end - overlap_region_start
                
                # Linear weights for this chunk: from 1.0 (start) to 0.0 (end)
                weights_this = torch.linspace(1.0, 0.0, overlap_size,
                                              device=device,
                                              dtype=x0.dtype)[None, :, None, None]

                # Get overlapping region from this chunk
                local_start_in_chunk = overlap_start_lat
                local_end_in_chunk = local_start_in_chunk + overlap_size
                x0_overlap = x0[:, local_start_in_chunk:local_end_in_chunk, :, :]

                # Add weighted contribution from this chunk
                buf[:, overlap_region_start:overlap_region_end, :, :] += \
                    weights_this * x0_overlap
                weight_sum[overlap_region_start:overlap_region_end] += weights_this.squeeze()

                # Add weighted contribution from next chunk
                next_x0 = x0_preds[chunk_idx + 1]
                
                # Linear weights for next chunk: from 0.0 (start) to 1.0 (end)
                weights_next = torch.linspace(0.0, 1.0, overlap_size,
                                               device=device,
                                               dtype=next_x0.dtype)[None, :, None, None]

                # Get overlapping region from next chunk (starts from frame 0)
                next_x0_overlap = next_x0[:, :overlap_size, :, :]

                # Add weighted contribution from next chunk
                buf[:, overlap_region_start:overlap_region_end, :, :] += \
                    weights_next * next_x0_overlap
                weight_sum[overlap_region_start:overlap_region_end] += weights_next.squeeze()

        # Non-overlapping region after overlap (only for last chunk)
        if chunk_idx == num_chunks - 1:
            post_overlap_start = start + overlap_start_lat
            post_overlap_end = min(chunk_end, T_total_lat)
            
            # Check if there's a non-overlapping region after overlap
            if chunk_idx > 0:
                prev_chunk_end = starts[chunk_idx - 1] + T_win_lat
                post_overlap_start = max(post_overlap_start, prev_chunk_end)
            
            if post_overlap_start < post_overlap_end:
                local_start = post_overlap_start - start
                buf[:, post_overlap_start:post_overlap_end, :, :] = \
                    x0[:, local_start:local_start + (post_overlap_end - post_overlap_start), :, :]
                weight_sum[post_overlap_start:post_overlap_end] = 1.0

    # Normalize by weight sum (should be 1.0 everywhere, but handle edge cases)
    return buf / weight_sum[None, :, None, None].clamp(min=1.0)


class WanT2VLong(WanT2V):
    """Long video generation by extending WanT2V with SyncTweedies.

    Inherits all model components (T5, VAE, DiT) from WanT2V and adds a
    new `generate_long` method that can produce videos longer than the
    model's native frame limit by using overlapping temporal windows and
    predicted-x0 weighted averaging at each UniPC ODE step.
    """

    def generate_long(
        self,
        input_prompt: str,
        size: tuple = (832, 480),
        window_size: int = 81,
        multiplier: int = 2,
        overlap_start: int = 40,
        max_steps: int = 50,
        shift: float = 5.0,
        guide_scale: float = 6.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
    ):
        """Generate a long video using multiplier-based chunking with weighted averaging.

        Uses the same UniPC ODE solver as text2video.py, but generates multiple
        overlapping chunks and applies linear weighted averaging in overlap regions.

        Args:
            input_prompt: Text prompt describing the video content.
            size: (width, height) of the output video in pixels.
            window_size: Frames per temporal window (model's native limit).
                         Must satisfy 4n+1.
            multiplier: Number of chunks to generate. total_frames is calculated as
                        window_size + (multiplier - 1) * (window_size - overlap_start).
            overlap_start: Frame index within a chunk where overlap starts (0-indexed).
                           Must be < window_size.
            max_steps: Number of UniPC ODE denoising steps.
            shift: Noise schedule shift parameter (same as text2video).
            guide_scale: Classifier-free guidance scale.
            n_prompt: Negative prompt. Falls back to config default if empty.
            seed: RNG seed. -1 means random.
            offload_model: Offload DiT to CPU between steps to save VRAM.

        Returns:
            torch.Tensor: [3, total_frames, H, W] video in [-1, 1] range,
                          or None for non-rank-0 processes.
        """
        # ------------------------------------------------------------------ #
        # 1.  Validate arguments                                               #
        # ------------------------------------------------------------------ #
        assert (window_size - 1) % 4 == 0, (
            f"window_size must satisfy 4n+1, got {window_size}")
        assert multiplier >= 1, f"multiplier must be >= 1, got {multiplier}"
        assert 0 < overlap_start < window_size, (
            f"overlap_start must be in (0, window_size), got {overlap_start}")

        # ------------------------------------------------------------------ #
        # 2.  Latent geometry                                                  #
        # ------------------------------------------------------------------ #
        # Get window starts and total latent frames
        starts, T_total_lat, T_win_lat, overlap_start_lat = _get_window_starts_multiplier(
            window_size, multiplier, overlap_start, self.vae_stride[0])
        
        # Calculate total_frames from latent space
        total_frames = (T_total_lat - 1) * self.vae_stride[0] + 1
        
        # Validate that total_frames satisfies 4n+1
        assert (total_frames - 1) % 4 == 0, (
            f"Calculated total_frames={total_frames} (from T_total_lat={T_total_lat}) "
            f"must satisfy 4n+1. Try adjusting overlap_start or multiplier.")
        
        H_lat = size[1] // self.vae_stride[1]
        W_lat = size[0] // self.vae_stride[2]
        C = self.vae.model.z_dim  # 16

        # seq_len for one window (all windows are the same size)
        seq_len = math.ceil(
            (H_lat * W_lat) / (self.patch_size[1] * self.patch_size[2]) *
            T_win_lat / self.sp_size) * self.sp_size

        # ------------------------------------------------------------------ #
        # 3.  Window positions                                                 #
        # ------------------------------------------------------------------ #
        num_windows = len(starts)
        overlap_lat = T_win_lat - overlap_start_lat
        logging.info(
            f"[LongVideo] window_size={window_size}, multiplier={multiplier}, "
            f"overlap_start={overlap_start}px ({overlap_start_lat}lat), "
            f"total_frames={total_frames}, num_windows={num_windows}, "
            f"T_total_lat={T_total_lat}, T_win_lat={T_win_lat}, "
            f"overlap={overlap_lat}lat ({overlap_lat * self.vae_stride[0]}px)"
        )
        sys.stdout.flush()  # Force flush for immediate output
        logging.info(f"[LongVideo] window starts (latent): {starts}")
        sys.stdout.flush()  # Force flush for immediate output

        # ------------------------------------------------------------------ #
        # 4.  Seed & RNG                                                       #
        # ------------------------------------------------------------------ #
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # ------------------------------------------------------------------ #
        # 5.  Text encoding                                                    #
        # ------------------------------------------------------------------ #
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # ------------------------------------------------------------------ #
        # 6.  Initialise full-video latent buffer with random noise            #
        #     (this is x_T on the ODE trajectory, same as text2video.py)      #
        # ------------------------------------------------------------------ #
        video_latent = torch.randn(
            C, T_total_lat, H_lat, W_lat,
            dtype=torch.float32,
            device=self.device,
            generator=seed_g,
        )

        # ------------------------------------------------------------------ #
        # 7.  UniPC scheduler — same as text2video.py                         #
        # ------------------------------------------------------------------ #
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            max_steps, device=self.device, shift=shift)
        timesteps = sample_scheduler.timesteps

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # ------------------------------------------------------------------ #
        # 8.  ODE denoising loop (SyncTweedies)                               #
        # ------------------------------------------------------------------ #
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Use tqdm with explicit settings for real-time output
            # mininterval=0 ensures immediate updates, file=sys.stderr is tqdm's default
            for step_idx, t in enumerate(
                    tqdm(timesteps, desc="SyncTweedies", file=sys.stderr, mininterval=0, miniters=1)):

                    # Initialize scheduler step_index if needed (matches scheduler.step() behavior)
                    if sample_scheduler.step_index is None:
                        sample_scheduler._init_step_index(t)

                    # Get sigma from scheduler's current step_index
                    # This matches what scheduler.convert_model_output() will use
                    sigma = sample_scheduler.sigmas[sample_scheduler.step_index].item()

                    # ---------------------------------------------------------- #
                    # A. Extract overlapping windows from current ODE latent     #
                    # ---------------------------------------------------------- #
                    windows = _extract_windows(video_latent, starts, T_win_lat)

                    # ---------------------------------------------------------- #
                    # B. Get x̂₀ per window via model + Tweedie (BATCHED)         #
                    #    x̂₀ = x_t - σ · v_pred     (flow matching Tweedie)      #
                    #    Model accepts List[Tensor] and batches internally      #
                    # ---------------------------------------------------------- #
                    self.model.to(self.device)
                    # Replicate timestep for batch processing: [t] -> [t, t, t, ...]
                    timestep = torch.stack([t] * len(windows))  # [B] shape

                    # Batch forward pass: pass all windows as list, model batches internally
                    # Context needs to be replicated for each window
                    # Model expects context to be a list of tensors, same for all windows
                    vc_batch = self.model(
                        windows, t=timestep,
                        context=context, seq_len=seq_len)  # Returns List[Tensor]
                    vu_batch = self.model(
                        windows, t=timestep,
                        context=context_null, seq_len=seq_len)  # Returns List[Tensor]

                    # CFG and Tweedie per window
                    x0_preds = []
                    for i, window in enumerate(windows):
                        vc = vc_batch[i]
                        vu = vu_batch[i]
                        vp = vu + guide_scale * (vc - vu)
                        # Tweedie: x̂₀ = x_t - σ · v_pred
                        x0_preds.append((window - sigma * vp).detach())

                    del vc_batch, vu_batch

                    if offload_model:
                        self.model.cpu()
                        torch.cuda.empty_cache()

                    # ---------------------------------------------------------- #
                    # C. Weighted average x̂₀ in overlap regions → x̂₀_full      #
                    # ---------------------------------------------------------- #
                    x0_full = _aggregate_x0_weighted(
                        x0_preds, starts, T_win_lat,
                        T_total_lat, overlap_start_lat,
                        C, H_lat, W_lat,
                        self.device,
                    ).to(video_latent.dtype)

                    del x0_preds, windows

                    # ---------------------------------------------------------- #
                    # D. UniPC ODE step on the full latent using refined x̂₀_full  #
                    #    Use step_with_refined_x0 to directly use averaged x0     #
                    # ---------------------------------------------------------- #
                    video_latent = sample_scheduler.step_with_refined_x0(
                        x0_full.unsqueeze(0),
                        t,
                        video_latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0].squeeze(0)

                    del x0_full

            # -------------------------------------------------------------- #
            # 9.  VAE decode                                                   #
            # -------------------------------------------------------------- #
            if self.rank == 0:
                logging.info("[LongVideo] Decoding with VAE ...")
                sys.stdout.flush()  # Force flush for immediate output
                video = self.vae.decode([video_latent])[0]
            else:
                video = None

        del video_latent, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return video

    def generate_long_cached(
        self,
        input_prompt: str,
        size: tuple = (832, 480),
        window_size: int = 81,
        multiplier: int = 2,
        overlap_start: int = 40,
        max_steps: int = 50,
        shift: float = 5.0,
        guide_scale: float = 6.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
        soft_blend: bool = False,
    ):
        """Generate a long video using Tweedie Caching: sequential window processing with x0 caching.

        Processes windows sequentially, reusing cached x0 predictions from previous windows
        for overlapping regions. More efficient than SyncTweedies but processes windows
        sequentially instead of in parallel.

        Args:
            input_prompt: Text prompt describing the video content.
            size: (width, height) of the output video in pixels.
            window_size: Frames per temporal window (model's native limit).
                         Must satisfy 4n+1.
            multiplier: Number of chunks to generate. total_frames is calculated as
                        window_size + (multiplier - 1) * (window_size - overlap_start).
            overlap_start: Frame index within a chunk where overlap starts (0-indexed).
                           Must be < window_size.
            max_steps: Number of UniPC ODE denoising steps.
            shift: Noise schedule shift parameter (same as text2video).
            guide_scale: Classifier-free guidance scale.
            n_prompt: Negative prompt. Falls back to config default if empty.
            seed: RNG seed. -1 means random.
            offload_model: Offload DiT to CPU between steps to save VRAM.

        Returns:
            torch.Tensor: [3, total_frames, H, W] video in [-1, 1] range,
                          or None for non-rank-0 processes.
        """
        # ------------------------------------------------------------------ #
        # 1-7. Same setup as generate_long (validation, geometry, encoding, etc.)
        # ------------------------------------------------------------------ #
        assert (window_size - 1) % 4 == 0, (
            f"window_size must satisfy 4n+1, got {window_size}")
        assert multiplier >= 1, f"multiplier must be >= 1, got {multiplier}"
        assert 0 < overlap_start < window_size, (
            f"overlap_start must be in (0, window_size), got {overlap_start}")

        starts, T_total_lat, T_win_lat, overlap_start_lat = _get_window_starts_multiplier(
            window_size, multiplier, overlap_start, self.vae_stride[0])
        
        total_frames = (T_total_lat - 1) * self.vae_stride[0] + 1
        
        assert (total_frames - 1) % 4 == 0, (
            f"Calculated total_frames={total_frames} (from T_total_lat={T_total_lat}) "
            f"must satisfy 4n+1. Try adjusting overlap_start or multiplier.")
        
        H_lat = size[1] // self.vae_stride[1]
        W_lat = size[0] // self.vae_stride[2]
        C = self.vae.model.z_dim  # 16

        seq_len = math.ceil(
            (H_lat * W_lat) / (self.patch_size[1] * self.patch_size[2]) *
            T_win_lat / self.sp_size) * self.sp_size

        num_windows = len(starts)
        overlap_lat = T_win_lat - overlap_start_lat
        logging.info(
            f"[LongVideo-Cached] window_size={window_size}, multiplier={multiplier}, "
            f"overlap_start={overlap_start}px ({overlap_start_lat}lat), "
            f"total_frames={total_frames}, num_windows={num_windows}, "
            f"T_total_lat={T_total_lat}, T_win_lat={T_win_lat}, "
            f"overlap={overlap_lat}lat ({overlap_lat * self.vae_stride[0]}px)"
        )
        sys.stdout.flush()
        logging.info(f"[LongVideo-Cached] window starts (latent): {starts}")
        sys.stdout.flush()

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        video_latent = torch.randn(
            C, T_total_lat, H_lat, W_lat,
            dtype=torch.float32,
            device=self.device,
            generator=seed_g,
        )

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            max_steps, device=self.device, shift=shift)
        timesteps = sample_scheduler.timesteps

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # ------------------------------------------------------------------ #
        # 8.  ODE denoising loop (Tweedie Caching)                            #
        #     Process windows sequentially: each window independently         #
        # ------------------------------------------------------------------ #
        
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Process each window sequentially (OUTER LOOP)
            pbar_windows = tqdm(
                enumerate(starts), 
                total=len(starts),
                desc="TweedieCaching (Windows)", 
                file=sys.stderr, 
                mininterval=0, 
                miniters=1,
                position=0,
                leave=True)
            
            for win_idx, start in pbar_windows:
                win_end = start + T_win_lat
                
                # Create a new scheduler for this window (each window processes independently)
                # Each window needs its own scheduler with fresh timesteps
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    max_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps  # Get fresh timesteps for this window
                
                # Get initial timestep sigma (first sigma, which is the highest noise level)
                # This is used to re-inject noise into the overlap region
                sigma_initial = sample_scheduler.sigmas[0].item()
                
                # Determine which part to reuse from cache (for later windows)
                if win_idx == 0:
                    # First window: compute everything
                    reuse_start_in_window = None
                    reuse_end_in_window = None
                else:
                    # Later windows: reuse overlapping part from cache
                    prev_start = starts[win_idx - 1]
                    prev_end = prev_start + T_win_lat
                    # Overlap region: from max(start, prev_start + overlap_start_lat) to min(win_end, prev_end)
                    overlap_start_global = max(start, prev_start + overlap_start_lat)
                    overlap_end_global = min(start + T_win_lat, prev_end)
                    
                    if overlap_start_global < overlap_end_global:
                        # Reuse overlapping part (in current window's local coordinates)
                        reuse_start_in_window = overlap_start_global - start
                        reuse_end_in_window = overlap_end_global - start
                    else:
                        # No overlap
                        reuse_start_in_window = None
                        reuse_end_in_window = None
                
                # Generate window_latent:
                # - Overlap region: re-inject noise into previous window's denoised latent
                # - Non-overlap region: pure gaussian noise
                window_latent = torch.randn(
                    C, T_win_lat, H_lat, W_lat,
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g,
                )
                
                # For later windows, re-inject noise into the overlap region
                # This ensures continuity: overlap region starts from previous window's result
                # but with noise re-injected so it can be denoised again
                if win_idx > 0 and reuse_start_in_window is not None and reuse_end_in_window is not None:
                    # Get the final denoised latent from previous window
                    prev_window_final_latent = video_latent[:, overlap_start_global:overlap_end_global, :, :].clone()
                    
                    # Generate noise for the overlap region
                    overlap_noise = torch.randn(
                        C, reuse_end_in_window - reuse_start_in_window, H_lat, W_lat,
                        dtype=torch.float32,
                        device=self.device,
                        generator=seed_g,
                    )
                    
                    # Re-inject noise: x_t = (1 - sigma_t) * x_0 + sigma_t * noise
                    # This brings the denoised latent back to the initial noise level
                    window_latent[:, reuse_start_in_window:reuse_end_in_window, :, :] = \
                        (1 - sigma_initial) * prev_window_final_latent + sigma_initial * overlap_noise
                
                # Determine overlap region for caching (what to cache for next window)
                # This should match what the next window will reuse
                if win_idx < len(starts) - 1:
                    # There's a next window
                    next_start = starts[win_idx + 1]
                    # The overlap region that next window will reuse
                    # Next window's reuse region: from max(next_start, start + overlap_start_lat) to min(next_start + T_win_lat, start + T_win_lat)
                    next_overlap_start_global = max(next_start, start + overlap_start_lat)
                    next_overlap_end_global = min(next_start + T_win_lat, start + T_win_lat)
                    
                    if next_overlap_start_global < next_overlap_end_global:
                        # Cache the region that next window will reuse (in current window's local coordinates)
                        overlap_start_in_window = next_overlap_start_global - start
                        overlap_end_in_window = next_overlap_end_global - start
                    else:
                        overlap_start_in_window = None
                        overlap_end_in_window = None
                else:
                    overlap_start_in_window = None
                    overlap_end_in_window = None
                
                # Process all timesteps for this window (INNER LOOP)
                pbar_steps = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    desc=f"  Window {win_idx+1}/{len(starts)} (Steps)",
                    file=sys.stderr,
                    mininterval=0,
                    miniters=1,
                    position=1,
                    leave=False)
                
                for step_idx, t in pbar_steps:
                    # Initialize scheduler step_index for this timestep
                    # step_with_refined_x0 will also check, but we need sigma before calling it
                    if sample_scheduler.step_index is None:
                        sample_scheduler._init_step_index(t)
                    else:
                        # Ensure step_index matches current timestep
                        # This is important because scheduler maintains state across steps
                        current_step_idx = sample_scheduler.index_for_timestep(t)
                        if sample_scheduler.step_index != current_step_idx:
                            sample_scheduler._step_index = current_step_idx
                    
                    # Get sigma from scheduler's current step_index
                    sigma = sample_scheduler.sigmas[sample_scheduler.step_index].item()
                    
                    # Use t.item() as dict key since tensor is not hashable
                    t_key = t.item() if isinstance(t, torch.Tensor) else t
                    
                    # Prepare x0 for this window at this timestep
                    # Always compute entire window first
                    self.model.to(self.device)
                    timestep_tensor = torch.stack([t])
                    
                    vc = self.model(
                        [window_latent], t=timestep_tensor,
                        context=[context], seq_len=seq_len)[0]
                    vu = self.model(
                        [window_latent], t=timestep_tensor,
                        context=[context_null], seq_len=seq_len)[0]
                    
                    vp = vu + guide_scale * (vc - vu)
                    
                    # Compute x0 from velocity
                    # Tweedie: x̂₀ = x_t - σ · v_pred
                    window_x0 = (window_latent - sigma * vp).detach()
                    
                    del vc, vu
                    
                    del vp
                    
                    # UniPC ODE step on this window's latent
                    window_latent = sample_scheduler.step_with_refined_x0(
                        window_x0.unsqueeze(0),
                        t,
                        window_latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0].squeeze(0)
                    
                    del window_x0
                    
                    if offload_model:
                        self.model.cpu()
                        torch.cuda.empty_cache()
                
                # Write completed window back to full video latent
                video_latent[:, start:win_end, :, :] = window_latent
                
                # Debug: Save each window's denoising result as a separate video
                if self.rank == 0:
                    try:
                        import os
                        from wan.utils.utils import cache_video
                        
                        # Decode this window's latent to video
                        window_video = self.vae.decode([window_latent])[0]
                        
                        # Save to debug_windows directory in current working directory
                        debug_dir = os.path.join(os.getcwd(), "debug_windows")
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        # Save window video using cache_video (same format as main video)
                        window_filename = f"window_{win_idx}.mp4"
                        window_path = os.path.join(debug_dir, window_filename)
                        cache_video(
                            tensor=window_video[None],
                            save_file=window_path,
                            fps=self.config.sample_fps,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1)
                        )
                        logging.info(f"[Debug] Saved window {win_idx} video to {window_path}")
                        del window_video
                    except Exception as e:
                        logging.warning(f"[Debug] Failed to save window {win_idx} video: {e}")
                
                del window_latent

            # -------------------------------------------------------------- #
            # 9.  VAE decode                                                   #
            # -------------------------------------------------------------- #
            if self.rank == 0:
                logging.info("[LongVideo-Cached] Decoding with VAE ...")
                sys.stdout.flush()
                video = self.vae.decode([video_latent])[0]
            else:
                video = None

        del video_latent, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return video
