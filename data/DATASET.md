# NIH Malaria Dataset Analysis

## Source
- **Dataset**: NIH Malaria Cell Images
- **URL**: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
- **Total images**: 27,558
- **Classes**: Parasitized (13,779), Uninfected (13,779)
- **Balance**: Perfect 50/50

## Image Characteristics
- **Resolution range**: 40×46 to 385×394 pixels
- **Mean resolution**: 133×132 pixels
- **Pixel intensity**: 0-214 (grayscale), mean=114.3, std=78.0

## Observations from Manual Inspection
- Black backgrounds dominant
- 2-3 blurry parasitized cells observed (out of 40 sampled)
- No partial cells
- No visible debris or artifacts

## Known Issues
- **Extreme resolution variability**: 9× size difference between smallest and largest
- **Small images**: 157 images < 80px (investigate)
- **Pixel max = 214**: Not full uint8 range, suggests preprocessing or brightness limitation

## Preprocessing Decisions (LOCKED)

### Input Resolution
- **Target**: 224×224 pixels
- **Method**: Aspect-ratio preserving resize + symmetric black padding
- **Rationale**: Minimizes distortion while maintaining standard ViT input size

### Small Image Handling
- **Count**: 157 images with min(H, W) < 80px (0.57% of dataset)
- **Decision**: **KEEP, DO NOT FILTER**
- **Rationale**: 
  - Creates natural failure modes for uncertainty estimation
  - Tests model robustness on low-resolution inputs
  - Expected to trigger higher predictive uncertainty (validation target)
  - Removing them introduces artificial data cleanliness
- **Tracking**: Tagged as low-resolution samples for failure analysis

### Normalization
- **Method**: Dataset-specific mean and standard deviation
- **Computation**: Train split only (prevent data leakage)
- **Rationale**: Blood smear images differ from ImageNet natural images

### Patch Size
- **Size**: 16×16 pixels (fixed)
- **Patches per image**: 14×14 = 196 patches
- **Revisit condition**: Only if attention analysis proves insufficient parasite detail capture

## Limitations
- Single lab source → limited stain variation
- Controlled conditions → may not generalize to field settings
- Blob-like cell crops → occlusion/debris not well-represented