# SeedVR Studio

SeedVR Studio upscales visual media. This language distinguishes collections of independent images from groups of frames within one source.

## Language

**Image batch**:
A collection of independent still images selected for processing together with shared upscaling settings.
_Avoid_: Batch (without qualification), image sequence, temporal batch

**Output root**:
A parent directory chosen as the destination for image-batch results.
_Avoid_: Batch output folder, source folder

**Batch output folder**:
The directory containing the results belonging to one image batch.
_Avoid_: Output root, source folder

**Temporal batch**:
A group of frames processed together during upscaling of one source. Its size counts frames, not files in an image batch.
_Avoid_: Image batch, file batch
