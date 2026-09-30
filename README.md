If newly importing into system, please consider:
1. Putting the "mode" to "full" first before doing anything

## Runtime / Scalability Testing

For runtime benchmarking, the goal is to measure how PGD and GSR-PGD scale as the number of evaluated images increases.

### Recommended Test Sizes

Use the following evaluation sizes:

- 10 images
- 50 images
- 100 images
- 250 images
- 500 images
- 1000 images

Use 1 image only for debugging, not for final runtime measurements.

### Configuration

For runtime-only experiments, do not include the tuning split.

Set:

```python
TUNING_SIZE = 0
MAX_IMAGES = 10
TOTAL_IMAGES = MAX_IMAGES + TUNING_SIZE