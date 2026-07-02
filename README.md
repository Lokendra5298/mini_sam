# MiniSAM: A Small SAM-Style Model From Scratch

MiniSAM is a small educational implementation of a SAM-style promptable segmentation model built from scratch in PyTorch.

The goal of this project is to understand the internal architecture of Segment Anything-style models by implementing the main components step by step:

- ViT-style image encoder
- Point, box, and mask prompt encoder
- Two-way mask decoder
- Multiple mask tokens
- IoU prediction head
- Focal loss, dice loss, and IoU prediction loss
- SAM-style best-mask training objective

Project blog:

```text
https://lokendra5298.github.io/mini_sam/


## 1. What MiniSAM Learns

MiniSAM is not trained as a normal image-to-mask segmentation model.

Instead, it learns:

```text
Image + Prompt → Object Mask
```

The prompt can be a point or a box. During training, the box and positive point prompts are generated automatically from the ground-truth mask.

---

## 2. Architecture Overview

```text
Input Image
   ↓
ViT-style Image Encoder
   ↓
Image Embedding

Point / Box / Mask Prompt
   ↓
Prompt Encoder
   ↓
Prompt Embedding

Image Embedding + Prompt Embedding
   ↓
Two-Way Mask Decoder
   ↓
Mask Tokens + IoU Token
   ↓
Hypernetwork Mask Head
   ↓
Final Segmentation Masks
```

Main components:

| Component      | Role                                                        |
| -------------- | ----------------------------------------------------------- |
| Image Encoder  | Converts input image into dense visual tokens               |
| Prompt Encoder | Converts point, box, and mask prompts into embeddings       |
| Mask Decoder   | Uses two-way attention to connect prompt and image features |
| Mask Tokens    | Produce multiple candidate masks                            |
| IoU Head       | Predicts quality score for each mask                        |
| Losses         | Focal loss, dice loss, and IoU prediction loss              |

---

## 3. Final Validation Results

MiniSAM was trained from scratch on the Oxford-IIIT Pet segmentation dataset.

| Metric               |  Value |
| -------------------- | -----: |
| Validation IoU       | 0.8949 |
| Validation Dice      | 0.9440 |
| Precision            | 0.9447 |
| Recall               | 0.9442 |
| Pixel Accuracy       | 0.9028 |
| Oracle Best-Mask IoU | 0.8954 |

The selected-mask IoU is very close to the oracle best-mask IoU. This means the IoU prediction head learned to select almost the best mask candidate.

---

## 4. Project Structure

```text
mini_sam/
├── dataset.py
├── losses.py
├── train.py
├── models/
│   ├── common.py
│   ├── image_encoder.py
│   ├── prompt_encoder.py
│   ├── mask_decoder.py
│   └── mini_sam.py
├── scripts/
│   ├── download_oxford_pet.py
│   ├── evaluate.py
│   ├── plot_training_log.py
│   └── visualize_predictions.py
├── tests/
│   ├── test_random_image_encoder.py
│   ├── test_random_prompt_encoder.py
│   ├── test_random_mask_decoder.py
│   ├── test_random_mini_sam.py
│   ├── test_random_losses.py
│   └── test_dataset.py
└── docs/
    ├── index.html
    ├── architecture.html
    ├── training.html
    ├── results.html
    └── assets/
```

---

## 5. Installation

```bash
git clone https://github.com/Lokendra5298/mini_sam.git
cd mini_sam

pip install -r requirements.txt
```

---

## 6. Download Dataset

```bash
python scripts/download_oxford_pet.py
```

This creates:

```text
data/oxford_pet/processed/train/images
data/oxford_pet/processed/train/masks
data/oxford_pet/processed/val/images
data/oxford_pet/processed/val/masks
```

---

## 7. Run Random Tensor Tests

Before training, test every module independently:

```bash
export PYTHONPATH=$PWD

python tests/test_random_image_encoder.py
python tests/test_random_prompt_encoder.py
python tests/test_random_mask_decoder.py
python tests/test_random_mini_sam.py
python tests/test_random_losses.py
python tests/test_dataset.py
```

---

## 8. Train MiniSAM

```bash
python train.py \
  --data_root data/oxford_pet/processed \
  --out_dir runs/minisam_oxford_pet_v2_sam_loss_ep50_bs32 \
  --device_id 0 \
  --epochs 50 \
  --batch_size 32 \
  --workers 8 \
  --lr 1e-4
```

The training script saves:

```text
best.pt       # best validation loss
best_iou.pt   # best validation IoU
last.pt       # final epoch checkpoint
```

---

## 9. Evaluate

```bash
python scripts/evaluate.py \
  --data_root data/oxford_pet/processed \
  --split val \
  --checkpoint runs/minisam_oxford_pet_v2_sam_loss_ep50_bs32/last.pt \
  --device_id 0 \
  --batch_size 32 \
  --workers 8
```

---

## 10. Visualize Predictions

```bash
python scripts/visualize_predictions.py \
  --data_root data/oxford_pet/processed \
  --split val \
  --checkpoint runs/minisam_oxford_pet_v2_sam_loss_ep50_bs32/last.pt \
  --out_dir runs/minisam_oxford_pet_v2_sam_loss_ep50_bs32/visuals_final \
  --device_id 0 \
  --num_samples 32
```

---

## 11. Training Lesson

The first version used average loss over all mask candidates. This caused validation loss to increase because weak extra mask candidates were punished even when one candidate mask was good.

The corrected version uses SAM-style best-mask training:

```text
j* = argmin_j L_mask(j)

L = L_mask(j*) + L_iou
```

This improved validation IoU and made training behavior cleaner.

---

## 12. MiniSAM vs Original SAM

MiniSAM is not a full reproduction of Meta's Segment Anything Model.

| Feature       | Original SAM                 | MiniSAM                                 |
| ------------- | ---------------------------- | --------------------------------------- |
| Goal          | General segmentation         | Educational from-scratch implementation |
| Image size    | Large resolution             | 256 × 256                               |
| Image encoder | Large ViT                    | Small ViT-style encoder                 |
| Dataset       | Very large segmentation data | Oxford-IIIT Pet                         |
| Prompt types  | Point, box, mask             | Point, box, mask                        |
| Decoder       | Two-way transformer          | Small two-way transformer               |
| Output        | Multiple masks + IoU         | Three masks + IoU                       |
| Purpose       | Zero-shot segmentation       | Learning SAM internals                  |

---

## 13. Blog Pages

* Home: `https://lokendra5298.github.io/mini_sam/`
* Architecture: `https://lokendra5298.github.io/mini_sam/architecture.html`
* Training Details: `https://lokendra5298.github.io/mini_sam/training.html`
* Results: `https://lokendra5298.github.io/mini_sam/results.html`

---

## 14. Author

Lokendra Kumar
GitHub: [Lokendra5298](https://github.com/Lokendra5298)
