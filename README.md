# Wan Long Video Generation Experiments

This repository contains experimental results and code for long video generation using Wan2.1-T2V models.

## Contents

- `wan/long_video.py`: Implementation of long video generation methods (SyncTweedies and Tweedie Caching)
- `video_synctweedies_protocol.md`: Protocol documentation for SyncTweedies method
- `run_long_video_experiments.sh`: Script for running experiments
- `prompts_30.txt`: Collection of prompts used in experiments
- `output_*/`: Generated video results organized by timestamp

## Methods

1. **SyncTweedies**: Overlapping temporal windows with predicted x0 weighted averaging
2. **Tweedie Caching**: Sequential window processing with x0 caching

## Usage

See individual files for usage instructions.

## Results

Video results are stored in `output_*/` directories, named with timestamps and experiment parameters.
