# Image Generation Metrics

This directory provides a distributed evaluator for:

- PickScore
- Aesthetic Score
- HPSv2
- ImageReward
- CLIP Score
- DINOv2 FID
- DINO feature diversity
- DINOv2 recall-to-teacher

The evaluator requires CUDA. The default launcher uses eight GPU processes; set
`--nproc_per_node` in `calculate_metrics.sh` to match the available GPUs.
Missing metric models are downloaded automatically and cached under
`METRICS_MODEL_ROOT`.

## Configuration

Set the following paths at the top of `calculate_metrics.sh`:

- `METRICS_MODEL_ROOT`: persistent cache directory for metric models.
- `TEACHER_SAMPLES_DIR`: directory containing teacher/reference images.
- `STUDENT_SAMPLES_DIR`: directory containing generated student images.
- `PROMPT_JSON_PATH`: JSON file mapping student image filenames to prompts.

## Expected Sample Layout

Both sample directories must be flat. Images in nested directories are not
discovered.

```text
<samples_root>/
├── teacher/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── student/
    ├── 000000.jpg
    ├── 000001.jpg
    ├── ...
    └── prompts.json
```

Teacher and student filenames do not need to match. The teacher directory only
needs reference images; it does not need a prompt JSON file.

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, and
`.webp`. DINO metrics require at least two images in each directory.

The prompt JSON keys are filenames relative to `STUDENT_SAMPLES_DIR`, not
absolute paths:

```json
{
  "000000.jpg": "a photo of a cat",
  "000001.jpg": "a photo of a dog"
}
```

Keep one JSON entry for every student image and avoid placing unrelated images
in the directory. Prompt-based metrics process files listed in the JSON, while
DINO metrics process every supported image directly inside the directory.

The TRD inference launchers produce this student layout automatically. Point
`STUDENT_SAMPLES_DIR` to their `SAVE_DIR` and `PROMPT_JSON_PATH` to the
generated `prompts.json`.

## Environment

The metric dependencies are intentionally installed in a separate environment
from training:

```bash
conda create -n trd_eval python=3.10 -y
conda activate trd_eval
```

Install a CUDA-compatible PyTorch build, then install:

```bash
pip install numpy scipy pillow tqdm torchvision
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
pip install transformers==4.37.2
pip install huggingface_hub==0.23.0
```

## Run

From the repository root:

```bash
bash calc_metrics/calculate_metrics.sh
```

Additional Python arguments can be appended to the launcher:

```bash
bash calc_metrics/calculate_metrics.sh \
  --dinov2_recall_k 5 \
  --dinov2_distance_chunk_size 512
```

`dinov2_recall_k` must be smaller than the number of student images. Results
are printed as a summary to standard output; the evaluator does not write a
result file.

## DINO Metric Notes

DINO feature diversity groups student images by exact prompt text, calculates
the mean pairwise cosine distance within each group, and averages over prompts
with at least two images.

DINOv2 recall-to-teacher measures the fraction of teacher features covered by
the student feature manifold. The manifold radius around each student feature
is defined by its `k`-th nearest student neighbor.
