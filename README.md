# Update 23 January 2026
Dataset is not yet available, as our University finalizes the security and legal matters to data sharing. Apology for any inconvenience caused. Stay tuned for any updates.

# TFD68
TFD68: A Fully Annotated Pose-Invariant Thermal Facial Dataset with Occlusions and Visual Pairs Using 68-Point Landmarks

# Installing Dependencies
Install `requirements.txt` using
```bash
pip install -r requirements.txt 
```
# Dataset Access
Use this link to download dataset used in this project and place in "input" folder:

https://compvis.site.hw.ac.uk/dataset/tfd-thermal-facial-dataset/

Use this link to download the pre-trained model:

```text
tfd68_unet/
├─ input/
│  ├─ tfd68/
│  └─ tfd68.json
```

# Code Usage
1) Place data and annotations as above.
2) Run prepare_dataset.py
   ```python
   python -m dataset.prepare_dataset
   ```
3) Run train.py
   ```python
   python -m train
   ```
4) Once trained, run test.py
   ```python
   python -m test
   ```
5) Results will be in /out


# Referencing
If you make use of this code or TFD68 in your work. Please cite this work as:

Yean Chun Ng, Alexander G. Belyaev, F. C. M. Choong, Shahrel Azmin
Suandi, Joon Huang Chuah, and Bhuvendhraa Rudrusamy. 2025. TFD68: A
Fully Annotated Thermal Facial Dataset with 68 Landmarks, Pose Variations,
Per-Pixel Thermal Maps, Visual Pairs, Occlusions, and Facial Expressions.
In SIGGRAPH Asia 2025 Technical Communications (SA Technical Communications
’25), December 15–18, 2025, Hong Kong, China. ACM, New York, NY,
USA, 4 pages. https://doi.org/10.1145/3757376.3771410
