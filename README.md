# Healthcare Analytics Project — imported working project


## Setup and demonstration (Windows PowerShell)

Use Python 3.12 for compatibility with the original environment.

```powershell
cd "C:\NADIA\Sem7\Health care\HEALTH CARE ANALYTICS PROJECT\ckd_progression_mm_gnn\ckd_progression_mm_gnn"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Select the Stage 1 High-Risk preset, choose the first Stage 1 scan, and click **Run Multimodal Prediction (Scan + Labs)**. Sample clinical values are in examples/sample_input.csv. Existing demonstration screenshots and evaluation reports are in examples/ and the application directory. Scores may vary by dependency version; the screenshots are original recorded outputs, not newly measured results.

## Included files and structure

- ckd_progression_mm_gnn/ckd_progression_mm_gnn/: complete Python application, main.py, run_ablation.py, train_vision_classifier.py and tests.
- Within that application: data/ (clinical CSVs and documented preprocessing), model/ (vision, fusion and GNN modules), graph/, train/, eval/, cache/ (pre-extracted visual features), .streamlit/config.toml, requirements.txt, presentation_assets/ and original README.md.
- Trained artifacts: best_model.pt, best_mm_model.pt, ckpt_full_mm.pt, ckpt_ablated_mm.pt, vision_stage_classifier.pt and mm_eval_summary.pt.
- ckd_progression_mm_gnn/dataset/: Stage0-4 through Stage4-4 scan datasets, annotations and original dataset provenance/readme files.
- examples/: demonstration input, recorded screenshots and evaluation output.

No separate database or login is required; clinical data and scan files are local. Important functions already include comments/docstrings in the imported source.

## Validation and optional training

Run from the application directory with the environment above:

```powershell
.\.venv\Scripts\python.exe -m unittest test_pipeline test_mm_pipeline
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe run_ablation.py
```

Training commands overwrite checkpoints in this imported copy. They are not required to launch the saved-model demonstration.

## Existing portability and reproduction details

The application source is preserved byte-for-byte. The dataset loader discovers the bundled ../dataset directory before legacy fallback paths, so always launch from the application directory. Some original scripts retain absolute paths from the development machine: train_vision_classifier.py uses C:\healthcare analytics with images-stage0 through images-stage4 folder names, and presentation scripts may refer to the original author's folders. Set these paths in this copy before rerunning those optional scripts.

The vision encoder requests torchvision ConvNeXt-Tiny pretrained weights on first use (internet access or an existing Torch cache is needed). Its existing implementation falls back to random backbone weights if that load fails. The bundled feature-cache keys may contain original absolute scan paths; unmatched entries fall back to zero vectors in the existing loader. Consequently, the preserved source may not reproduce identical numerical visual-feature results after relocation. Original evaluation reports are supplied as recorded outputs, not accuracy guarantees.

The clinical longitudinal trajectories are generated from the bundled baseline dataset. Scans and clinical rows are paired by CKD stage rather than patient identity, as documented in data/loader.py.

## Import verification

All 930 imported files matched their source SHA-256 hashes. The copied project passed all 12 existing unit tests (test_pipeline and test_mm_pipeline), using Python 3.12.14 and torch 2.14.1+cpu from the existing environment with bytecode writes disabled. The web interface was not exercised interactively. requirements-original-environment.txt records installed package versions for reference; requirements.txt in the application directory is the primary dependency list.

