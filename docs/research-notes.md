# Research notes: what the literature and past RSNA winners say we can still gain (2026-09-29)

Scope: RSNA Knee Abnormality Detection (12 binary findings, macro AUC; 4,407 train studies with LLM-read report labels, 58 radiologist-labelled gold studies; our 2.5D CoAtNet-2 @ 280 px, 15 slices x 6 plane/sequence slots, per-label attention, 5-fold, LB 0.912; public forks 0.940-0.944; leaders 0.95-0.959). 23 days, one RTX 5090 + ~30 Kaggle GPU-h/week.

Conventions used below:
- **[M]** = number measured and reported by the cited source; **[O]** = opinion / inference (the source's or ours).
- "Applicability" = can we test it inside 3 weeks with our compute: **high / medium / low**.
- Access caveat: every Kaggle-hosted page (competition writeups, discussion threads) is JavaScript-rendered and returned an empty shell to `WebFetch`, `curl` and the `r.jina.ai` proxy, and Kaggle's internal discussion API returned 404 anonymously. Where a Kaggle writeup is the primary source, the details below come from the team's GitHub repo, a peer-reviewed summary, or a third-party summary, and that is what is cited. The Kaggle URL is still given so you can open it in a browser.

---

## 1. Winning solutions of the most similar Kaggle competitions

### 1.1 Summary table

| Source | Method (input / backbone / head) | Measured delta or score | Applicability |
|---|---|---|---|
| RSNA 2024 Lumbar Spine, 1st place ("Avengers"). Kaggle writeup (not fetchable): https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution ; summarised in arXiv 2512.12887 App. E3: https://arxiv.org/pdf/2512.12887 | Localize-then-classify: 3D localisation of level coordinates, then classifier on multi-view crops = "2.5D stacks of sagittal and axial slices"; 2D backbone ConvNeXt-S or EfficientNet-V2-S; bidirectional LSTM + attention-MIL fusing slices and views; auxiliary attention losses; ensemble across backbones and folds [M: architecture as described by the review] | No numeric ablations available from an accessible source | medium (we already have attention-MIL over slots; the transferable idea is *view-specific crops + aux losses*) |
| RSNA 2024 Lumbar Spine, 2nd place (Bartley / Ian Pan / Yuji Ariyasu), GitHub README: https://github.com/brendanartley/RSNA-2024-Competition | Sagittal-only 2-stage; stage 1 uses "eight middle frames"; stage 2 T1 "middle 24 frames", T2 "between 5 - 16 frames"; encoder + LSTM + attention pooling; pseudo-labelling as "0.5*loss_with_labels + 0.5*loss_with_pseudolabels"; label denoising via "confident learning"; "manifold mixup to the embeddings", sequence-order flip, left/right recombination; "nine rotations provide TTA" [M: design; no deltas] | Negative result: "Axial T2 pipeline that improved my CV by 0.01-0.02, but it was not strong enough to improve our overall team CV" [M] | **high** for pseudo-label 50/50 loss and confident-learning label pruning (both cheap); medium for manifold mixup |
| RSNA 2023 Abdominal Trauma, 1st place (Nischay Dhankhar et al.), GitHub: https://github.com/Nischaydnk/RSNA-2023-1st-place-solution | 96 equidistant slices reshaped to (32, 3, 384, 384) i.e. 2.5D triplets of adjacent slices; CoaT-Lite Medium/Small + GRU and EfficientNet-V2-S + GRU; BCE + auxiliary Dice segmentation head; soft slice labels = patient label x organ visibility; max over slice logits; 4-fold GroupKFold [M] | Auxiliary segmentation loss "+0.01 to +0.03 boost" (log-loss metric) [M]; best single 4-fold OOF 0.326 vs best ensemble "0.31x" [M] | low for seg-aux (no masks for knee); medium for the *soft-target* idea |
| RSNA 2023 Abdominal Trauma, 2nd place (Theo Viel), GitHub: https://github.com/TheoViel/kaggle_rsna_abdominal_trauma | 2D `maxvit_tiny_tf_512` with "3 adjacent frames used as channels"; crop models `coatnet_1_rw_224` + RNN on "11 frames sampled uniformly in the organ"; heavy augs + cutmix p=0.5 (p=1.0 for crop models) [M] | ConvNeXt-v2 alone private 0.39 vs MaxViT + CoatNet-RNN 0.35 (lower is better); full ensemble "3x2D models, 8x 2.5D models" private 0.35 [M] | medium (backbone family diversity matters more than count) |
| RSNA 2023 top-7 patterns, yu4u slides: https://speakerdeck.com/yu4u/rsna-2023-abdominal-trauma-detection-fan-sheng-hui | Every top solution: segmentation-based crop -> 2.5D CNN + RNN; 3rd place: "2.5D + RNN model", multiple backbones/necks/crop sizes with weighted averaging; "Pipeline similarities to RSNA 2022" [M: description] | none | medium |
| RSNA 2022 Cervical Spine, 1st place (Qishen Ha). Kaggle writeup (not fetchable): https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution ; summarised in arXiv 2512.12887 App. E1 and https://arxiv.org/html/2601.15235v4 | Stage 1: 3D U-Net (resnet18d / effv2s) at 128^3 trained on 87 masks; Stage 2: 2.5D CNN-LSTM, "15 slices are sampled along the z-axis and concatenated with neighboring slices and segmentation masks", EfficientNet-V2-S or ConvNeXt backbone, patient-level model on 7x15 slices; ensemble over backbones and folds [M: description] | Top-8 algorithms of the challenge: mean AUC 0.96 (95% CI 0.95-0.96) on the challenge test set [M] https://pubmed.ncbi.nlm.nih.gov/38169426/ | medium (15-slice sampling is what we already do) |
| RSNA 2022, 3rd place (Darragh Hanley), GitHub: https://github.com/darraghdog/RSNA22 | "random sequence of 96 dicoms and converts them to 32 * 3-channel images"; 2D CNN + 1D RNN; slice-level pseudo labels from study labels x vertebra ratio; multi-seed ("3 random seeds", 6 and 4 seeds for the fracture models); seed-averaged boxes [M] | no deltas reported | medium (multi-seed is cheap on a 5090) |
| RSNA 2021 Brain Tumor (MGMT), 1st place (Firas Baba), code: https://github.com/FirasBaba/rsna-resnet10 ; details as summarised in https://www.mdpi.com/2078-2489/13/3/124 and https://arxiv.org/pdf/2201.04416 | 3D ResNet-10, 64 slices centred by the "central image trick" (slice with largest brain cross-section), 256 px, BCE, Adam, 15 epochs, batch 8 [M as summarised] | 5-fold CV AUC ~0.62 [M as summarised]; the whole task sat near the noise floor (leaderboard ~0.6) | low (different regime: label was essentially unpredictable) |
| RSNA 2019 Intracranial Hemorrhage top-5 (context for 2.5D + sequence heads), arXiv 2205.07556: https://arxiv.org/pdf/2205.07556 | "Nearly all the top solutions rely on 2D convolutional networks and sequential models (Bidirectional GRU or LSTM)"; ensembles of 7-31 models [M] | Single end-to-end transformer model 0.04368 log-loss vs 1st place 20-model ensemble 0.04383; 3-model ensemble 0.04345 [M] | medium: one well-trained sequence model can match a 20-model ensemble |
| Slice-fusion ablation (2D foundation-model review, Table 6, task T5): https://arxiv.org/pdf/2512.12887 | Same DINOv3 features, different fusion heads | Avg-pool 0.958, max-pool 0.946, median 0.944, LSTM 0.903 (5.5 M params), Transformer 0.950 (7.0 M), query-attention 0.962 (0.001 M) AUC [M] | **high** (negative evidence: do not spend the 3 weeks on GRU/LSTM heads; a light attention pool is enough) |

### 1.2 Training tricks with measured deltas (from any source found)

| Trick | Source | Measured delta | Applicability |
|---|---|---|---|
| Auxiliary dense head (segmentation) | RSNA 2023 1st, https://github.com/Nischaydnk/RSNA-2023-1st-place-solution | "+0.01 to +0.03" (log-loss) [M] | low (no knee masks) |
| Pseudo-labels mixed 50/50 with hard labels; confident-learning denoising | RSNA 2024 2nd, https://github.com/brendanartley/RSNA-2024-Competition | used, no delta [M: usage] | high |
| TTA (rotations x9) | same | used, no delta | medium (inference budget 9 h) |
| TTA on CheXpert ensemble | Pham et al., https://arxiv.org/pdf/1911.06475 Table 4 | removing TTA "only decreases the mean AUC by 0.003" [M] | low priority |
| SWA (weight averaging for 10 epochs after convergence) | Izmailov et al. 2018, Table 2: https://arxiv.org/pdf/1803.05407 | ImageNet top-1: ResNet-50 76.15 -> 76.97, ResNet-152 78.31 -> 78.94, DenseNet-161 77.65 -> 78.44 ("0.6-0.9%") [M] | low-medium; our own fold-0 SWA test was flat within run-to-run noise (~0.005) [internal measurement, no URL] |
| EMA of weights | Knee repo with 0.942 public LB uses "EMA and cosine schedule": https://github.com/Hisernberg/knee | no delta reported | medium (near-free) |
| Multi-seed averaging | RSNA 2022 3rd: https://github.com/darraghdog/RSNA22 ; RSNA 2019 ensembles of 7-31 models: https://arxiv.org/pdf/2205.07556 | no isolated delta | high (cheap; variance reduction) |
| Distillation of an ensemble into one net | Noothout et al., J. Med. Imaging 2022: https://pmc.ncbi.nlm.nih.gov/articles/PMC9142841/ | Student "reached the performance of the ensembles they were derived from" (e.g. SegTHOR aorta 0.94 vs ensemble 0.93 Dice; ACDC LV 0.96 vs 0.97), "5 to 8 times faster"; students "significantly outperformed the same network trained with reference segmentation" [M] | high (this is the "multi-teacher pseudo-label" mechanism the leaders credit) |

### 1.3 What winners say did NOT help [M unless noted]
- Adding an axial pipeline that gave +0.01-0.02 on one member's CV did not move the team ensemble (RSNA 2024 2nd): https://github.com/brendanartley/RSNA-2024-Competition
- Heavy sequence heads: LSTM fusion 0.903 vs plain average pooling 0.958 AUC on the same features (review Table 6): https://arxiv.org/pdf/2512.12887
- ConvNeXt-v2 alone was clearly worse (0.39) than MaxViT/CoAtNet-based models (0.35) in RSNA 2023 2nd: https://github.com/TheoViel/kaggle_rsna_abdominal_trauma
- Gated attention pooling lost to max-pooling at the knee-competition scale (18th-place blog, single model): https://huggingface.co/blog/bishnoiyash/rsna-competetion
- Fitting blend weights on OOF/gold: "per-target weights fitted by 5-fold CV scored -0.0008" vs equal weights; blend tuning on a label-table consensus "-0.0054" (knee competitor, public LB 0.935): https://github.com/ShivenKhurana1/rsna-knee-abnormality

---

## 2. Peer-reviewed knee-MRI deep learning: which plane/sequence for which finding

### 2.1 Table

| Source | Method | Measured result | Applicability |
|---|---|---|---|
| MRNet, Bien et al. 2018 PLOS Med: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699 | 1,130 / 120 / 120 exams; AlexNet per slice -> max-pool over slices -> per-plane model; three planes (sagittal T2, coronal T1, axial PD) combined by **logistic regression on the plane-level probabilities** | AUC abnormality 0.937 (0.895-0.980), ACL 0.965 (0.938-0.993), meniscus 0.847 (0.780-0.914) [M]; external (Croatia, 183 exams) ACL 0.824 -> 0.911 after retraining [M]; the paper reports axial PD contributed most for abnormality and meniscus, coronal T1 for ACL [M, as extracted] | high: per-plane models + late logistic fusion is a cheap, proven baseline for a "plane-expert" arm |
| ELNet, Tsai et al. MIDL 2020: https://arxiv.org/html/2005.02706v3 (PMLR: https://proceedings.mlr.press/v121/tsai20a.html) | *Single plane per task*: meniscus -> coronal, ACL -> axial, abnormality -> axial (MRNet); sagittal T1 on KneeMRI; 211k params, multi-slice normalisation + BlurPool | MRNet-val AUC: meniscus 0.904 (vs MRNet 0.826), ACL 0.960 (0.956), abnormality 0.941 (0.936) [M]; KneeMRI 5-fold mean 0.913 [M] | medium: shows a *single* well-chosen plane can beat 3-plane fusion for meniscus |
| MRPyrNet, Dunnhofer et al. MIDL 2021: https://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf | Feature-pyramid + pyramidal detail pooling added to MRNet / ELNet (small lesions) | MRNet -> +MRPyrNet: ACL 0.955 -> 0.976, meniscus 0.843 -> 0.889; ELNet -> +MRPyrNet: ACL 0.940 -> 0.960, meniscus 0.869 -> 0.895; KneeMRI ACL 0.902 -> 0.914 [M]; ablation: FPN alone 0.960/0.859, PDP alone 0.971/0.855, both 0.972/0.879 (ACL/meniscus) [M] | medium: argues for higher effective resolution / multi-scale features on small structures (meniscus, cartilage) |
| TripleMRNet plane study, Yang et al. QIMS 2025: https://qims.amegroups.org/article/view/137915/html (PubMed https://pubmed.ncbi.nlm.nih.gov/40606344/) | MRNet data; AlexNet features from 1/2/3 planes, feature-level concat; 7 plane combinations; reports ACC/F1 (no AUC) | ACL ACC: sag 0.892, cor 0.842, ax 0.867, sag+ax 0.917, all three 0.925 [M]. Meniscus ACC: sag 0.633, cor 0.750, ax 0.775, all three 0.783 [M]. Authors: "For meniscal tear detection, the axial plane markedly outperformed the other two planes ... and the sagittal plane exhibited the poorest performance" [M]; three-plane beat every single/two-plane combination for both tasks [M] | high: check that our 6 slots always include an axial fluid-sensitive series; consider plane-specific arms for meniscus |
| Multi-plane hybrid fusion (J. AI, Dergipark 2025): https://dergipark.org.tr/en/pub/jai/article/1926635 | Plane-specific encoders + fusion on MRNet | ACL best single plane sagittal AUC 0.976; meniscus sag 0.791 vs axial 0.826 [M]; fused ACL 0.949 [M]; "long-range sequential dependency modeling may provide limited benefit in plane-specific knee MRI analysis where diagnostically relevant findings are often localized within adjacent slices" [M: authors' conclusion] | high (supports 3-slice windows + attention rather than long RNNs) |
| Fritz et al. 2020 (3D CNN meniscus), as summarised in Frontiers 2021: https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.747217/full ; PubMed https://pubmed.ncbi.nlm.nih.gov/32170334/ | 3D CNN on knee MRI | AUC medial 0.882, lateral 0.781, any meniscus 0.961 [M] | medium: lateral meniscus is systematically the harder label; expect it to cap macro AUC |
| Rizk et al. 2021 (meniscal lesion detection, external validation): https://www.sciencedirect.com/science/article/pii/S1120179721000983 | Localisation + classification 3D CNNs | AUC medial 0.93, lateral 0.84; external 0.83 without and 0.89 with fine-tuning [M] | medium |
| Roblot et al. 2019: https://www.sciencedirect.com/science/article/pii/S2211568419300555 | Faster R-CNN on *sagittal* slices, horn detection then tear classification | AUC 0.92 [M] | low (detection-style) |
| Astuto et al. 2021 Radiology: AI (cartilage, BME, meniscus, ACL grading): https://pubs.rsna.org/doi/full/10.1148/ryai.2021200165 (abstract via https://pubmed.ncbi.nlm.nih.gov/34142088/) | 1,435 studies, 3D CNNs per tissue | Binary sensitivity 70-88%, specificity 85-89% [M]; cartilage AUC 0.93, meniscus 0.93, bone-marrow edema 0.83 [M, from the article summary returned by search] | medium: contusion/BME and cartilage are inherently lower-AUC labels |
| Multi-plane cross-attention fusion (knee competitor repo, CoPAS): https://github.com/shraddhauser/rsna-knee-detection | 16 slices per plane, 224 px, BCE + focal | no scores reported | low |

### 2.2 Evidence on per-plane models + late fusion vs one multi-plane model
- **Late fusion of per-plane models works and is simple**: MRNet's three plane models were combined with logistic regression and reached 0.965 ACL / 0.847 meniscus [M] (https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699).
- **Feature-level three-plane fusion beat every single plane** for both ACL and meniscus in TripleMRNet [M] (https://qims.amegroups.org/article/view/137915/html), but a **single, correctly chosen plane can beat the fused baseline** (ELNet coronal-only meniscus 0.904 vs MRNet fused 0.826 [M], https://arxiv.org/html/2005.02706v3). [O] Taken together: plane choice per label matters more than the fusion mechanism; our per-label attention over slots should be *checked* (attention maps / slot-dropout ablation) to confirm it is actually routing meniscus to axial/coronal and ACL to sagittal.
- Plane-to-finding priors supported by the sources above: sagittal for ACL (0.976 single-plane AUC, https://dergipark.org.tr/en/pub/jai/article/1926635; ACC 0.892 vs 0.842/0.867, https://qims.amegroups.org/article/view/137915/html); axial (and coronal) for meniscus (ACC 0.775 / 0.750 vs sagittal 0.633, same QIMS source; coronal-only ELNet 0.904). No accessible paper quantified planes for MCL, effusion, synovitis, Baker's cyst, contusion or PF OA; the mapping "coronal for collateral ligaments, axial for PF cartilage/effusion" is radiological convention, not a measured DL result [O].

---

## 3. Noisy / LLM-derived labels: cheap, measured remedies

| Source | Method | Measured effect | Applicability |
|---|---|---|---|
| CheXpert, Irvin et al. 2019, Table 3: https://arxiv.org/pdf/1901.07031 | Uncertain ("-1") labels handled as U-Ignore / U-Zeros / U-Ones / U-SelfTrained / U-MultiClass; DenseNet-121, 30-checkpoint ensembles | Per-label AUC: Atelectasis 0.818 / 0.811 / **0.858** / 0.833 / 0.821; Cardiomegaly 0.828 / 0.840 / 0.832 / 0.831 / **0.854**; Consolidation 0.938 / 0.932 / 0.899 / **0.939** / 0.937; Edema 0.934 / 0.929 / **0.941** / 0.935 / 0.928; Effusion 0.928 / 0.931 / 0.934 / 0.932 / **0.936** [M]. Best policy is **label-specific**; spread up to 0.047 (Atelectasis) and 0.040 (Consolidation, where U-Ones *hurt*) | **high**: our "not addressed in report" labels are exactly this "u" class; choose the policy per finding on OOF, not globally |
| Pham et al. 2019/2021, Table 1: https://arxiv.org/pdf/1911.06475 | Label-smoothing regularisation on uncertain labels: U-Ones -> U[0.55, 0.85], U-Zeros -> U[0, 0.3]; conditional training (CT) on the label hierarchy | Mean AUC (5 labels): U-Ignore 0.863, U-Zeros 0.858, U-Ones 0.860; +CT 0.872 / 0.877 / 0.872; U-Ones+LSR 0.874, U-Zeros+LSR 0.871; U-Ones+CT+LSR **0.894**, U-Zeros+CT+LSR 0.884 [M]; template-matching preprocessing +0.006, TTA removal -0.003, 6-model ensemble 0.940 [M] | **high**: soft targets for "not mentioned" labels are a one-line change; expected order +0.01 mean AUC on CXR |
| GPT-4o labels for upper-extremity radiographs, Eur Radiol Exp 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC13035556/ | "uncertain" -> positive (inclusive) vs -> negative (exclusive) | Macro AUC clavicle 0.80 vs 0.81, elbow 0.80 vs 0.80, thumb 0.76 vs 0.78; "No significant differences ... (p >= 0.15)" [M]; LLM label accuracy 98.1-99.0% but it under-detected uncertainty (e.g. 10.5% manual vs 6.4% GPT-4o for elbow) [M] | high (says: when the LLM is accurate, uncertainty policy barely matters; the bigger risk is *missed* uncertainty) |
| Open-source LLM labels vs CheXpert labeler for CXR classifiers, J Imaging Inform Med 2025: https://pubmed.ncbi.nlm.nih.gov/40325326/ | Train classifiers on Llama-3 CoT labels vs CheXpert-labeler labels | Classifier AUC 0.96 (pneumothorax) / 0.84 (rib fracture) against human annotations vs 0.91 / 0.73 when scored against CheXpert labels; LLM sensitivity for fracture 95% vs CheXpert labeler 51% [M, from abstract] | medium: better report readers move the *evaluation*, and noisier labels "make it harder to generalize" |
| Co-teaching, Han et al. NeurIPS 2018, Table 5: https://arxiv.org/pdf/1804.06872 | Two nets exchange small-loss samples | CIFAR-10 test acc, Standard -> Co-teaching: sym-20% 76.25 -> 82.32; sym-50% 48.87 -> 74.02; pair-45% 49.50 -> 72.62 [M]; at 20% noise F-correction (84.55) was better [M] | low-medium: gains are large only at >=40% noise; our label noise is likely 10-20% |
| Confident learning, Northcutt et al. JAIR 2021, Table 2: https://arxiv.org/pdf/1911.00068 | Estimate the noisy/true joint from OOF predicted probabilities, prune likely-mislabelled examples, retrain | CIFAR-10 ResNet-50 acc: 20% noise baseline 78.4 -> CL 91.1; 40% noise 60.2 -> 86.7; Co-teaching 81.2 / 62.9 [M]; "Data-centric AI methods (CL, INCV, Mixup) outperform model-centric methods (SCE-Loss, MentorNet, Co-Teaching, S-Model)" [M: authors' summary] | **high**: model-agnostic, uses OOF predictions we already have; used by RSNA 2024 2nd place |
| Generalised cross-entropy (Lq / truncated Lq), Zhang & Sabuncu 2018, Table 1 (last block, CIFAR-100 uniform noise 0.2/0.4/0.6/0.8): https://arxiv.org/pdf/1805.07836 | Loss between CE and MAE | CCE 58.72 / 48.20 / 37.41 / 18.10 -> Lq 66.81 / 61.77 / 53.16 / 29.16; TruncLq 67.61 / 62.64 / 54.04 / 29.60 [M] | low: multi-class results; for binary BCE the analogue is mild label smoothing |
| Risk-aware robust learning on MedMNIST (2026): https://arxiv.org/html/2604.23875 | CE vs GMM filter / Co-teaching / DivideMix / UNICON at 0/20/40% symmetric noise | DermaMNIST 40%: baseline BAC 0.643, Co-teaching 0.650, UNICON 0.732; PathMNIST 40%: baseline 0.868, Co-teaching 0.870, UNICON 0.842 [M] | low: on medical data the sophisticated methods are inconsistent and Co-teaching is ~flat |
| RSNA 2024 2nd place: https://github.com/brendanartley/RSNA-2024-Competition | "confident learning" removal of labels whose predictions "deviated significantly from ground truth"; pseudo labels at 0.5 loss weight | no isolated delta | high |
| Knee competitor (public LB 0.942): https://github.com/Hisernberg/knee | "masked, confidence-weighted BCE" with "fold-calibrated report labels"; protocol embeddings (plane, fluid-sensitive, fat-sat, depth, series slot) | no isolated delta | high (already close to what we do; the *fold-calibrated* confidence weight is the new bit) |

[O] Synthesis: the cheapest measured wins are (a) per-label choice among ignore / zero / one / soft for "not mentioned" (CheXpert: up to 0.04-0.05 AUC on individual labels), (b) label smoothing of the uncertain class (Pham: +0.01-0.03 mean AUC), (c) confident-learning pruning or down-weighting using our own OOF (large on CIFAR, adopted by a top Kaggle team). Co-teaching / robust losses have weak evidence on medical data.

---

## 4. Ensembling theory: rank vs probability, blend weights, AUC noise floors

### 4.1 Table

| Source | Point | Measured / stated | Applicability |
|---|---|---|---|
| Hanley & McNeil 1982 formula as documented at https://metricgate.com/docs/hanley-mcneil-roc-comparison/ (also https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf) | SE of a single AUC | Var(A) = [A(1-A) + (n+ - 1)(Q1 - A^2) + (n- - 1)(Q2 - A^2)] / (n+ n-), Q1 = A/(2-A), Q2 = 2A^2/(1+A) [M: formula] | high (see noise-floor table below) |
| DeLong small-sample behaviour, arXiv 2208.01614: https://arxiv.org/pdf/2208.01614 | DeLong variance + logit CI | Accurate CIs "even when the total sample size is as small as 63 in one-sample setting and 56 in two-sample setting" [M, as summarised]; DeLong variance is "always positively biased" (conservative) https://www.sciencedirect.com/science/article/abs/pii/S0167865523003598 | high |
| Forecast-combination puzzle: Claeskens et al. 2016 https://www.sciencedirect.com/science/article/abs/pii/S0169207016000327 ; Smith & Wallis 2009 and review by Wang et al. 2022 Sec. 2.6: https://arxiv.org/pdf/2205.04216 | Estimated "optimal" weights vs equal weights | Because weights are estimated (random), the combination "will be biased ... and its variance will be larger than in the fixed-weight case"; "there is no guarantee that the 'optimal' forecast combination will be better than the equal-weights case" [M: theorem-level statement]. Review: "simple averaging often empirically dominates sophisticated weighting schemes that should (asymptotically) be superior"; Smith & Wallis show the "estimation cost of weighted averages when the optimal weights are close to equality" [M: quotes] | high: with 12 labels x ~6 arms and n=58 gold, fitted weights are expected to lose |
| Knee competitor measurement (public LB 0.935): https://github.com/ShivenKhurana1/rsna-knee-abnormality | Fitted vs equal blend weights, OOF at n=4,349 | "per-target weights fitted by 5-fold CV scored -0.0008" vs equal; adding a "fourth decorrelated family" "+0.0010, 95% CI [+0.0001, +0.0019]"; blend tuning on a label-table consensus "-0.0054" [M] | **high** (same competition, same n) |
| Knee competitor (public LB 0.942): https://github.com/Hisernberg/knee | "Folds and arms are blended by equal-weight average ranks" [M: design] | - | high |
| Kaggle Ensembling Guide (MLWave mirror): https://evah.github.io/2017-02-05/ensemble/ | When rank averaging helps | Rank averaging fixes the case where "Not all predictors are perfectly calibrated: they may be over- or underconfident"; plain averaging "works well for a wide range of problems ... and metrics (AUC, ...)" [O: practitioner guidance, no numbers] | high (rank-average when arms come from different training recipes / label sets) |
| Deep-ensemble aggregation study, arXiv 2607.05019 App. E: https://arxiv.org/pdf/2607.05019 | Logit vs probability vs temperature-calibrated probability averaging (accuracy metric) | "Logit averaging achieves the highest accuracy at every ensemble size. Probability averaging without calibration performs worst, and calibrated probability averaging falls between the two" [M: authors' result, CREMA-D, sizes 2-8] | medium: for AUC, averaging *logits* (or ranks) of differently-calibrated arms is the safer default |
| Blend-weight optimisers (Nelder-Mead / Powell on OOF): Kaggle notebook https://www.kaggle.com/code/ipythonx/optimizing-metrics-out-of-fold-weights-ensemble ; WMT21 QE paper https://arxiv.org/pdf/2109.08724 | Practice | Nelder-Mead on dev set "finds better optima on the development set while also leading to better performance on test" in the WMT21 report [M for that task]; no general sample-size rule found | medium |

### 4.2 AUC noise floors for our evaluation sets (computed here from the Hanley-McNeil formula above; [O] the independence assumption for the macro average is ours)

Per-label SE(AUC) at AUC = 0.90 (columns 0.85 / 0.90 / 0.93 / 0.95 in the full table):

| n | prevalence | n+ / n- | SE @0.85 | SE @0.90 | SE @0.93 | SE @0.95 |
|---|---|---|---|---|---|---|
| 58 | 0.05 | 3 / 55 | 0.142 | 0.121 | 0.104 | 0.089 |
| 58 | 0.10 | 6 / 52 | 0.101 | 0.086 | 0.074 | 0.063 |
| 58 | 0.20 | 12 / 46 | 0.073 | 0.062 | 0.053 | 0.045 |
| 58 | 0.35 | 20 / 38 | 0.059 | 0.049 | 0.042 | 0.035 |
| 300 | 0.10 | 30 / 270 | 0.045 | 0.038 | 0.033 | 0.028 |
| 300 | 0.35 | 105 / 195 | 0.025 | 0.021 | 0.018 | 0.015 |
| 1300 | 0.10 | 130 / 1170 | 0.022 | 0.018 | 0.016 | 0.013 |
| 1300 | 0.35 | 455 / 845 | 0.012 | 0.010 | 0.009 | 0.007 |

Macro-12 SE at AUC 0.90 with prevalences spread 0.05-0.50 and independent labels: **n=58: ~0.020; n=300: ~0.009; n=1300: ~0.004** (per-label SE ranges 0.042-0.121, 0.018-0.054, 0.009-0.026 respectively). [O] Practical reading: on gold-58 nothing smaller than ~0.04 macro (2 SE) is distinguishable; the public LB (if ~300 studies) resolves ~0.02; only the private set (~1,300) and our OOF on the 4.3k report-labelled studies resolve the 0.005-0.01 steps we are chasing. Rank differences of 0.912 vs 0.940 on the public LB are real; 0.940 vs 0.944 is at the edge.

---

## 5. RSNA knee competition forum (kaggle.com/competitions/rsna-knee-abnormality-detection/discussion)

Not retrievable: the discussion list, individual threads and writeups are JavaScript-rendered; `WebFetch`, `curl` (5.7 kB shell) and `r.jina.ai` returned no thread content, and the internal discussions API returned 404 without a session. Nothing after 2026-09-24 could be read from the forum itself. What *is* public elsewhere, with dates where known:

| Source | Claim | Measured? | Applicability |
|---|---|---|---|
| https://github.com/Hisernberg/knee (public LB 0.942, rank 304) | Four arms "ConvNeXt-T 320, CoAtNet-0 384/140 mm, EffNetV2-S 5-slice ASL, MaxViT-T 384/140 mm"; "fixed 140 mm field of view"; "up to K 3-slice windows"; protocol embeddings; optional within-series BiGRU + cross-series transformer; masked confidence-weighted BCE; EMA; equal-weight rank blend; author's estimate "Rank 1-3 (0.956+)", own target "0.943-0.945 (rank ~90-150)" | LB [M]; expectations [O] | high: physical-FOV resampling (140 mm) and protocol embeddings are two concrete differences from our fixed-pixel 280 px pipeline |
| https://github.com/ShivenKhurana1/rsna-knee-abnormality (public LB 0.935, ~rank 206/2,406) | DINOv2/CNN + BiGRU + CoAtNet blend; LB progression "0.910 -> 0.915 -> 0.930 -> 0.935"; fitted weights -0.0008; 4th decorrelated family +0.0010 [CI +0.0001, +0.0019]; Kaggle T4 fp16 254 ms/step vs bf16 797 ms/step | [M] | high (use fp16, not bf16, on T4) |
| https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection | CoAtNet 2.5-D attention-MIL with public "Raptor" checkpoints: gold-58 0.930, public LB 0.939 with fixed weights; DINOv2 baseline gold-58 0.813; public 0.939-0.941 notebooks add "DINO/RadImageNet chain, LB-probed weights" | [M] LB; "LB-probed" is the author's characterisation | medium: confirms the 0.94 forks are checkpoint ensembles with probed weights (private-LB risk for them) |
| https://huggingface.co/blog/bishnoiyash/rsna-competetion (18th of 792 on 2026-08-09, LB 0.903) | EfficientNet-B3, 288 px, 12 train / 32 eval slices, one series per plane, max-pool > gated attention, plain mean of 5 folds; label extractor accuracy 0.833 vs experts; CV 0.854 (gold-58 excluded) vs LB 0.903; opines an "intrinsic accuracy ceiling somewhere around 0.88 to 0.92" from report-image disagreement | LB/CV [M]; ceiling [O] | medium: early-season single-model reference point |
| https://github.com/raahimnawaz/rsna-knee-abnormality-detection (404 at fetch time) | multilingual report labeler | - | - |

No public source found that posts a *private*-relevant top-3 recipe; the leaders' "multi-teacher pseudo-labels" claim from our own notes could not be corroborated outside the forum.

---

## Ranked recommendations (expected macro-AUC gain is our estimate [O]; evidence in brackets)

1. **Fix the measurement first.** Use the Hanley-McNeil floors above: gold-58 macro SE ~0.02, so stop using it to *select* among variants; select on OOF over the ~4.3k report-labelled studies (SE ~0.004) and keep gold-58 only as a sanity check on label bias. Expected gain: 0 directly, but it prevents the 0.005-level "flat" conclusions (SWA, pseudo_v2, blends) from being noise misreads. [Formula: https://metricgate.com/docs/hanley-mcneil-roc-comparison/ ; small-n validity: https://arxiv.org/pdf/2208.01614]
2. **Per-label "not addressed" policy + label smoothing (1-2 GPU-days).** For each of the 12 findings choose among U-Ignore / U-Zeros / soft U[0,0.3] / soft U[0.55,0.85] by OOF AUC, then retrain the best arm. Expected +0.003 to +0.010 macro (CheXpert per-label spreads up to 0.047; Pham LSR +0.011 to +0.034 mean over baseline policies). [https://arxiv.org/pdf/1901.07031 Table 3; https://arxiv.org/pdf/1911.06475 Table 1; null result when LLM labels are ~99% accurate: https://pmc.ncbi.nlm.nih.gov/articles/PMC13035556/]
3. **Confident-learning pass on the report labels using current OOF probabilities** (hours): estimate per-label flip rates, drop or down-weight the top-confidence disagreements, retrain one arm and compare on OOF. Expected +0.002 to +0.008; larger if the report reader's error rate exceeds ~15% on some label. [CL vs baseline on CIFAR-10: https://arxiv.org/pdf/1911.00068 Table 2; used by RSNA 2024 2nd: https://github.com/brendanartley/RSNA-2024-Competition]
4. **Distil, don't just pseudo-label from equals.** The mechanism that closes ensemble-to-single gaps is a student trained on the *ensemble's* soft outputs, with hard labels mixed in (RSNA 2024 2nd: 0.5/0.5 loss; distilled students match their teacher ensembles in medical imaging). Our pseudo_v1/v2 were flat because the teachers were same-strength copies; the fix is a teacher that is measurably better on OOF (all arms + the public 0.94 checkpoints, rank-averaged), then a student at our resolution. Expected +0.003 to +0.010. [https://github.com/brendanartley/RSNA-2024-Competition ; https://pmc.ncbi.nlm.nih.gov/articles/PMC9142841/ ; RSNA 2019 single sequential model beating a 20-model ensemble: https://arxiv.org/pdf/2205.07556]
5. **Plane-expert arms with late (logistic) fusion, checked per label.** Train cheap single-plane models (axial fluid-sensitive for meniscus/effusion/synovitis; sagittal for ACL; coronal for MCL) and add them as arms, or at least verify by slot-dropout that our attention is using the right planes. Expected +0.002 to +0.008, mostly on meniscus and lateral-side labels. [MRNet LR fusion: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699 ; ELNet coronal-only meniscus 0.904 vs fused 0.826: https://arxiv.org/html/2005.02706v3 ; axial dominance for meniscus: https://qims.amegroups.org/article/view/137915/html]
6. **Ensemble by adding a decorrelated family, not by tuning weights.** Equal-weight *rank* (or logit) averaging; add one architecture family that is not CoAtNet/ConvNeXt (MaxViT-T or EffNetV2-S with 5-slice windows, as the 0.942 repo does). Expected +0.001 to +0.003 (measured +0.0010 [CI +0.0001, +0.0019] for a 4th family; fitted weights -0.0008). [https://github.com/ShivenKhurana1/rsna-knee-abnormality ; https://github.com/Hisernberg/knee ; theory: https://arxiv.org/pdf/2205.04216 Sec. 2.6]
7. **Physical-scale resampling (fixed 140 mm FOV) + protocol embeddings** as inputs to the existing head. Expected +0.001 to +0.005; unmeasured but it is the clearest recipe difference between our 0.912 and the 0.942 public repo, and it is cheap because the cache is rebuilt anyway. [https://github.com/Hisernberg/knee]
8. **Multi-seed x EMA on the final arms** (5090 idle time). Expected +0.001 to +0.003; SWA/EMA give ~0.6-0.9% top-1 on ImageNet and our own SWA test was within noise, so this is variance reduction, not a step change. [https://arxiv.org/pdf/1803.05407 Table 2]
9. **Do not spend time on**: GRU/LSTM slice heads (LSTM 0.903 vs avg-pool 0.958 on identical features, https://arxiv.org/pdf/2512.12887 Table 6; "limited benefit" of long-range modelling in knee MRI, https://dergipark.org.tr/en/pub/jai/article/1926635); co-teaching / robust losses (flat on medical data, https://arxiv.org/html/2604.23875); TTA beyond flips (-0.003 when removed on CheXpert, https://arxiv.org/pdf/1911.06475); fitting blend weights on gold-58 (Sec. 4).
10. **Resolution/backbone scaling** (280 -> 384 px, larger CoAtNet) is what 2023 winners used (384-512 px) but costs the most GPU time; expected +0.002 to +0.006 and only worth it after items 2-4 have changed the labels the model is trained on. [https://github.com/Nischaydnk/RSNA-2023-1st-place-solution ; https://github.com/TheoViel/kaggle_rsna_abdominal_trauma]

Sum of the plausible, non-overlapping gains from items 2-7 is roughly +0.01 to +0.03 macro AUC [O], i.e. from 0.912 to the 0.93-0.94 band; reaching the 0.955+ band would additionally need the leaders' teacher quality, which no public source documents.

## Addendum 2026-09-30 — two public write-ups on labels (read by a sub-agent)

Sources: https://huggingface.co/blog/bishnoiyash/rsna-competetion (rank 18 of 792 at the time,
0.903 LB; Qwen3.6-35B via Ollama as the labeler) and
https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection (gpt-5.4-mini labeler; its 0.939 LB /
0.930 gold-58 come from the public Raptor + residual-gated checkpoints, not from its own labels).

- Neither source found a large label lever: the repo's E2 (frozen features, head retrained on the
  best public label table) gained +0.006 macro, +0.067 on Synovitis, -0.015 on Lateral/PF OA; fused
  into the family it was +0.0016. Rank-ensembling label tables did not beat the best single table
  ("share lineage"). Gold agreement of tables: flight0234 hybrid v4 0.899, stevenleehans v4 0.893,
  pilkwang v2 0.870, Qwen2.5-32B 0.843 (our consensus_v1: 0.895).
- The repo's A02 is the useful fact: on gold-58, image models beat every report-label table on
  Effusion (0.95 vs 0.83), Synovitis (0.80 vs 0.69), Contusion (0.92 vs 0.82) and Medial OA (0.98
  vs 0.94) - "gold labels are image-based; report labels carry systematic (not random) noise on
  those findings". That is the rationale for the teach4 variant (src/label_variants.py) and
  supports uteacher.
- Unmentioned findings: hard 0 in the blog, 0.3-0.5 (usually 0.5) in the repo prompt; our
  consensus sits at ~0.18 with weight ~0.76. The blog's prompt fixes that moved gold accuracy
  0.828 -> 0.833: broader Synovitis (Hoffa, plica, impingement), traumatic-pattern rule for
  Contusion, avulsion/insufficiency in Fracture, sharper Effusion, current-only findings.
- Both authors say the report reader (~0.83-0.87 accuracy vs the radiologist) is the ceiling, the
  same conclusion as docs/methodology.md.
- The repo's own description of the Raptor family: CoAtNet-RMLP-2 @384, 2.5D triplets, per-finding
  attention MIL, 5 slots x 44-64 slices, 140 mm crop, per-series 2-98 % window, 5 seeds x 15
  epochs, window dropout. Our v5 arm copied the cache side of that and scored 0.900 LB (v6c
  0.912), so the cache is not the gap; seeds x dense sampling might be.
