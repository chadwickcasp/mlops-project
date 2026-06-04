# Computer Vision Nature Scavenger Hunt
An application to primarily test skills learned during the MLOps Zoomcamp course. It's also an attempt to make something interesting, fun, and potentially useful to get people outside into nature.

Cursor workflow automation (daily digests, weekly rule review, Monday suggestions): see [.cursor/workflow-review/README.md](.cursor/workflow-review/README.md).

## Desktop Segmentation GUI
- Uses MobileSAM (from Ultralytics) to produce image segmentation masks from a combination of point and box prompts that a user draws on an image.
- **Setup, run, and UI details:** see [image_segmentation/README.md](image_segmentation/README.md).

## Data Sourcing (GBIF)
- Sandbox workflow for species distribution data lives in `data_sourcing/gbif/`.
- **Setup, run, cache design, and outputs:** see [data_sourcing/gbif/README.md](data_sourcing/gbif/README.md).
