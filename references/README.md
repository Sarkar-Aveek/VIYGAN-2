# References and verified quotes

[← back to README](../README.md)

Every quote below was copied from the text of the paper named in its source line and checked against that text.
Page numbers are **PDF page numbers** (1 = first page of the file), not the journal's printed page numbers. Cuts
are marked [...]. Citation markers such as `[1]` are part of the original text. The PDFs are not redistributed here
(most are under arXiv's default licence); `download_papers.py` fetches the openly available ones into
`references/pdf/`.

Where each source is used: PSNR/SSIM and perceptual quality → [docs/05](../docs/05_evaluation.md#51-why-psnr-and-ssim-are-not-the-headline);
multi-date input and co-registration → [docs/06](../docs/06_geospatial_consistency.md); network and loss design →
[docs/03](../docs/03_models.md).

---

## 1. ESRGAN

Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Loy, C. C., Qiao, Y., Tang, X. "ESRGAN: Enhanced
Super-Resolution Generative Adversarial Networks." ECCV Workshops (PIRM), 2018.
arXiv: https://arxiv.org/abs/1809.00219 (v2) · PDF: https://arxiv.org/pdf/1809.00219

> However, these PSNR-oriented approaches tend to output over-smoothed results without sufficient high-frequency details, since the PSNR metric fundamentally disagrees with the subjective evaluation of human observers [1].

— Section 1 Introduction, p. 1 (source: https://arxiv.org/pdf/1809.00219)

> In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN [2] to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery.

— Abstract, p. 1 (source: https://arxiv.org/pdf/1809.00219)

> Second, we improve the discriminator using Relativistic average GAN (RaGAN) [2], which learns to judge "whether one image is more realistic than the other" rather than "whether one image is real or fake".

— Section 1 Introduction, p. 2 (source: https://arxiv.org/pdf/1809.00219)

> Contrary to the convention, we propose to use features before the activation layers, which will overcome two drawbacks of the original design. First, the activated features are very sparse, especially after a very deep network, as depicted in Fig. 6. [...] Second, using features after activation also causes inconsistent reconstructed brightness compared with the ground-truth image, which we will show in Sec. 4.4.

— Section 3.3 Perceptual Loss, p. 7 (source: https://arxiv.org/pdf/1809.00219). [...] = two sentences omitted (the
'baboon' example).

> SR algorithms are typically evaluated by several widely used distortion measures, e.g., PSNR and SSIM. However, these metrics fundamentally disagree with the subjective evaluation of human observers [1].

— Section 2 Related Work, p. 4 (source: https://arxiv.org/pdf/1809.00219)

## 2. SRGAN

Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang,
Z., Shi, W. "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network." CVPR, 2017.
arXiv: https://arxiv.org/abs/1609.04802 (v5) · PDF: https://arxiv.org/pdf/1609.04802

> The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution.

— Abstract, p. 1 (source: https://arxiv.org/pdf/1609.04802)

> However, while achieving particularly high PSNR, solutions of MSE optimization problems often lack high-frequency content which results in perceptually unsatisfying solutions with overly smooth textures (c.f . Figure 2).

— Section 2.2 Perceptual loss function, p. 5 (source: https://arxiv.org/pdf/1609.04802). "c.f ." is spaced that way
in the original.

> An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

— Abstract, p. 1 (source: https://arxiv.org/pdf/1609.04802)

> We have further shown that standard quantitative measures such as PSNR and SSIM fail to capture and accurately assess image quality with respect to the human visual system [56].

— Section 4 Discussion and future work, p. 7 (source: https://arxiv.org/pdf/1609.04802)

## 3. The Perception-Distortion Tradeoff

Blau, Y., Michaeli, T. "The Perception-Distortion Tradeoff." CVPR, 2018.
arXiv: https://arxiv.org/abs/1711.06077 (v4) · PDF: https://arxiv.org/pdf/1711.06077

> In this paper, we prove mathematically that distortion and perceptual quality are at odds with each other.

— Abstract, p. 1 (source: https://arxiv.org/pdf/1711.06077)

> As opposed to the common belief, this result holds true for any distortion measure, and is not only a problem of the PSNR or SSIM criteria.

— Abstract, p. 1 (source: https://arxiv.org/pdf/1711.06077)

> In fact, and perhaps counter-intuitively, algorithms that are superior in terms of perceptual quality, are often inferior in terms of e.g. PSNR and SSIM [3], [4], [5], [6], [7], [8], [9].

— Section 1 Introduction, p. 1 (source: https://arxiv.org/pdf/1711.06077)

> Namely, the lower the distortion of an algorithm, the more its distribution must deviate from the statistics of natural scenes. [...] Therefore, any distortion measure alone, is unsuitable for assessing image restoration methods.

— Section 7 Conclusion, p. 9 (source: https://arxiv.org/pdf/1711.06077). [...] = one sentence omitted.

## 4. LPIPS

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., Wang, O. "The Unreasonable Effectiveness of Deep Features as a
Perceptual Metric." CVPR, 2018.
arXiv: https://arxiv.org/abs/1801.03924 (v2) · PDF: https://arxiv.org/pdf/1801.03924

> Despite this, the most widely used perceptual metrics today, such as PSNR and SSIM, are simple, shallow functions, and fail to account for many nuances of human perception.

— Abstract, p. 1 (source: https://arxiv.org/pdf/1801.03924)

> We find that deep features outperform all previous metrics by large margins on our dataset.

— Abstract, p. 1 (source: https://arxiv.org/pdf/1801.03924)

## 5. Satlas super-resolution ("Zooming Out on Zooming In")

Wolters, P., Bastani, F., Kembhavi, A. "Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing."
arXiv preprint, 2023 (Allen Institute for AI).
arXiv: https://arxiv.org/abs/2311.18082 (v1) · PDF: https://arxiv.org/pdf/2311.18082

> Finding 1. PSNR and SSIM, the most widely used metrics, are insufficient and poorly correlate with human judgments. The correspondences between human preferences and those computed by the standard SUPER-RES metrics, PSNR, SSIM, and cPSNR, are very low, as shown in Figure 3.

— Section 3.1 Super-Resolution Human Judgement Dataset, p. 4 (source: https://arxiv.org/pdf/2311.18082)

> Although the two SUPER-RES outputs on the right are quite poor, with one being blurry and the other being sharply downsampled and disfiguring structures, they attain high PSNR and SSIM scores, far beyond an image output from an ESRGAN model which resembles the ground truth image far more closely to the human eye.

— Section 3 Metrics, p. 3 (source: https://arxiv.org/pdf/2311.18082)

> Note that the four images are ordered from best to worst based on human preference, and PSNR and SSIM increase in an opposite trend.

— Figure 2 caption, p. 4 (source: https://arxiv.org/pdf/2311.18082)

> We build a new dataset, S2-NAIP, consisting of 1.2 million pairs of low-resolution Sentinel-2 time series and highresolution NAIP images.

— Section 4.1 S2-NAIP Dataset, p. 5 (source: https://arxiv.org/pdf/2311.18082). "highresolution" is spelled that
way in the extracted text (the hyphen is at a line break in the PDF).

> Tiles correspond to Web-Mercator tiles at zoom level 17, i.e., the world is projected to a 2D plane and divided into a 2^17 X 2^17 grid, with each tile corresponding to a grid cell. Each NAIP tile corresponds to a time series of Sentinel-2 images, each having a resolution of 32x32 pixels.

— Section 4.1 S2-NAIP Dataset, p. 5 (source: https://arxiv.org/pdf/2311.18082). The exponent (2^17) is a superscript in the
PDF.

> Results from this study show that ESRGAN accurately generates the buildings from the high-resolution target image 94.70% of the time, while SR3 only 81.77% of the time.

— Section 5 Method Study, p. 6 (source: https://arxiv.org/pdf/2311.18082)

The paper does not state a rationale for using several Sentinel-2 images (the dataset pairs NAIP with a time series
by design); see the Notes at the end.

More from the same paper, used in docs/01 §1.5:

> Finding 3. Performance scales with dataset and model size.

— Section 4, p. 5 (source: https://arxiv.org/pdf/2311.18082)

> We define small, medium, and large versions of the ESRGAN with 17mil, 87mil, and 347mil parameters, respectively.

— Section 4, p. 5 (source: https://arxiv.org/pdf/2311.18082)

> Between the smallest and largest data splits, there is a ten point improvement in CLIPSCORE for the smallest model.

— Section 4, p. 5 (source: https://arxiv.org/pdf/2311.18082)

> Finding 4. GANs are capable super-resolution methods.

— Section 5, p. 5 (source: https://arxiv.org/pdf/2311.18082)

> Compared to training only on L2 loss, like SRCNN and HighResNet, the outputs are sharper and more similar to the target images, as shown in Figure 4.

— Section 5, p. 6 (source: https://arxiv.org/pdf/2311.18082)

> Between our ESRGAN and SR3 models, the ESRGAN is 200x faster.

— Section 5, p. 6 (source: https://arxiv.org/pdf/2311.18082)

> Finding 5. Diffusion models generate realistic but inaccurate images.

— Section 5 (Building Count Study), p. 6 (source: https://arxiv.org/pdf/2311.18082)

> Finding 7. Super-resolution outputs are ineffective for machine consumption.

> Via this preliminary study we find that using SUPER-RES outputs does not outperform inputting the original low-resolution images, as shown in Table 3.

— Section 7.1, p. 8 (source: https://arxiv.org/pdf/2311.18082)

From the project's README (https://github.com/allenai/satlas-super-resolution), which lists released models
"Varying number of input Sentinel-2 images (just RGB bands)" (1, 2, 4, 8, 16) and "Different Sentinel-2 bands used
as input (8 input images)" (10m, 20m, 60m):

> Most experiments utilize just TCI, but for non-TCI bands, the 16-bit source data is divided by 8160 and clipped to 0-1.

> Each image is 128x128px with RGB channels.

(The second sentence describes the NAIP target images of S2-NAIP.)

## 6. Real-ESRGAN

Wang, X., Xie, L., Dong, C., Shan, Y. "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic
Data." ICCV Workshops, 2021.
arXiv: https://arxiv.org/abs/2107.10833 (v2) · PDF: https://arxiv.org/pdf/2107.10833

> In addition, we employ a U-Net discriminator with spectral normalization to increase discriminator capability and stabilize the training dynamics.

— Abstract, p. 1 (source: https://arxiv.org/pdf/2107.10833)

> The UNet outputs realness values for each pixel, and can provide detailed per-pixel feedback to the generator. In the meanwhile, the U-Net structure and complicate degradations also increase the training instability. We employ the spectral normalization regularization [37] to stabilize the training dynamics. Moreover, we observe that spectral normalization is also beneficial to alleviate the over-sharp and annoying artifacts introduced by GAN training.

— Section 3.4 Networks and Training, pp. 5-6 (source: https://arxiv.org/pdf/2107.10833)

> A typical way of sharpening images is to employ a post-process algorithm, such as unsharp masking (USM). However, this algorithm tends to introduce overshoot artifacts. We empirically find that sharpening ground-truth images during training could achieve a better balance of sharpness and overshoot artifact suppression.

— Section 4.1 Datasets and Implementation ("Sharpen ground-truth images during training"), p. 6
(source: https://arxiv.org/pdf/2107.10833)

## 7. opensr-test

Aybar, C., Montero, D., Donike, S., Kalaitzis, F., Gómez-Chova, L. "A Comprehensive Benchmark for Optical Remote
Sensing Image Super-Resolution." IEEE Geoscience and Remote Sensing Letters, vol. 21, 5003105, 2024.
DOI: https://doi.org/10.1109/LGRS.2024.3401394 (CC BY-NC-ND 4.0). Not on arXiv (TechRxiv preprint
10.36227/techrxiv.171177496.69538893/v1).
PDF read: University of Valencia repository (published version), https://roderic.uv.es/bitstreams/6b2cfbe3-8b73-4852-aa72-dfa2c9ca0d23/download
(handle https://hdl.handle.net/10550/100298)

> Moreover, commonly used metrics often prioritize attributes that do not necessarily correspond to improvements in spatial resolution.

— Abstract, p. 1 (source: roderic.uv.es PDF above)

> While previous studies have often relied on pixelwise metrics, like peak signal-to-noise ratio (PSNR) and perceptual metrics, such as structural similarity index measure (SSIM) and learned perceptual image patch similarity (LPIPS) [8], these metrics have limitations. On the one hand, pixelwise metrics can be too sensitive to spatial translations and local reflectance disparities unrelated to the SR problem [9]. On the other hand, perceptual metrics are not specifically intended for SR.

— Section I Introduction, pp. 1-2 (source: roderic.uv.es PDF above)

> 1) Consistency Property: Any SR image, when degraded (downsampled) to the original LR spatial resolution, SRdown, must maintain consistent reflectance values and spatial alignment with its LR counterpart.

— Section III Protocol for Remote Sensing Image SR Evaluation, p. 3 (source: roderic.uv.es PDF above).
"SRdown" is SR with subscript "down" in the PDF.

> 2) Synthesis Property: Any SR image must improve the effective spatial resolution. In addition, the SR model must preserve the low-frequency details from the original LR image.

— Section III, p. 3 (source: roderic.uv.es PDF above)

> 3) Correctness Property: Any SR model must avoid hallucinations and omissions. Hallucinations are related to generating high-frequency features that do not align with the real ones present in the HR image. On the other hand, omissions denote the missing high-frequency features that the SR image failed to capture.

— Section III, p. 3 (source: roderic.uv.es PDF above)

> In the realm of consistency, our framework introduces three distinct metrics, which are delineated in purple in Fig. 2: reflectance, spectral, and spatial consistency. The core objective here is to evaluate the fidelity with which the SR image retains the intrinsic properties of the LR image.

— Section IV SR Quality Metrics, p. 3 (source: roderic.uv.es PDF above)

> Regarding correctness, the primary objective is to classify and quantify how much of the high-frequency representation is an improvement (SR close to HR), omissions (SR close to LR), or hallucinations (SR far from both LR and HR).

— Section IV SR Quality Metrics, p. 4 (source: roderic.uv.es PDF above)

> Lastly, in terms of correctness, the generative models (SR4RS and diffuser) demonstrate a higher incidence of hallucinations across all datasets (Fig. 3). These models tend to introduce artifacts that, while visually appealing, do not accurately correlate with the HR data.

— Section V Results, p. 5 (source: roderic.uv.es PDF above)

**How the spatial score is computed:** the published paper (p. 3) defines spatial consistency as "the mean absolute error
between the matching points identified by LightGlue in both SR and LR images". The opensr-test package README
(https://github.com/ESAOpenSR/opensr-test, README.md) says instead: "**spatial**: The spatial alignment between the SR
and LR images. By default, it uses Phase Correlation Coefficient (PCC)." The package version used here (1.3.3) follows
the README, so our `spatial` numbers are phase-correlation shifts. Also from the README (documentation, not the
paper):

> traditional evaluation metrics such as PSNR, LPIPS, and SSIM are not designed to assess SR performance. These metrics fall short, especially in conditions involving changes in luminance or spatial misalignments - scenarios frequently encountered in real world.

— README.md, Introduction (source: https://raw.githubusercontent.com/ESAOpenSR/opensr-test/main/README.md)

## 8. Multi-image SR for satellites

### 8a. PROBA-V (cPSNR)

Märtens, M., Izzo, D., Krzic, A., Cox, D. "Super-Resolution of PROBA-V Images Using Convolutional Neural Networks."
Astrodynamics 3, 387-402, 2019.
arXiv: https://arxiv.org/abs/1907.01821 (v1) · PDF: https://arxiv.org/pdf/1907.01821

> While SISR is a mathematically ill-posed problem, in which lost pixel-information is somewhat imputed to generate a plausible and pleasing visual outcome, challenges in MISR can have a different objective: provided the multiple images are not exactly identical, subpixel differences may allow to obtain actual information on the ground truth image. This makes the approach particularly attractive for EO satellites as it allows to enhance their payload capabilities keeping the information content of the newly created pixel values linked to real observations.

— Section 2.2 Multi image Super-resolution, p. 4 (source: https://arxiv.org/pdf/1907.01821)

> However, this human perception bias makes them less applicable to scientific EO tasks, where the visual representation might be less important than, for example, accurate pixel values.

— Section 4 Quality Metric, p. 9 (source: https://arxiv.org/pdf/1907.01821). "them" = image quality metrics such as
SSIM and VIF (previous sentence).

> Since we can compensate for a constant bias in intensity much more easily than for a reconstruction containing noise or troublesome features like clouds, we soften our quality metric by removing bias from SR before computing anything else.

— Section 4.1 Ranking Super-resolved Images, p. 10 (source: https://arxiv.org/pdf/1907.01821)

> With the cPSNR we introduced a new quality metric that is applicable for images which only have partial information (due to cloud coverage) and is less sensitive to pixel-shifts and constant biases in intensity opposed to the unmodified PSNR.

— Conclusion, p. 18 (source: https://arxiv.org/pdf/1907.01821)

(The shift tolerance is defined on p. 10 as a search over crops: "we look for a displacement of SR by at most 3 pixels
in each direction that minimizes the error".)

### 8b. HighRes-net

Deudon, M., Kalaitzis, A., Goytom, I., Arefin, M. R., Lin, Z., Sankaran, K., Michalski, V., Kahou, S. E., Cornebise,
J., Bengio, Y. "HighRes-net: Recursive Fusion for Multi-Frame Super-Resolution of Satellite Imagery." arXiv preprint,
2020.
arXiv: https://arxiv.org/abs/2002.06460 (v1) · PDF: https://arxiv.org/pdf/2002.06460

> One result of the generalized sampling theory is that we can go beyond the Nyquist limit of any individual uniform sample by interleaving several uniform samples taken concurrently. When an image is down-sampled to a lower resolution, its high-frequency details are lost permanently and cannot be recovered from any image in isolation. However, by combining multiple low-resolution images, it is possible to recover the original scene at a higher resolution.

— Section 1 Introduction, p. 2 (source: https://arxiv.org/pdf/2002.06460)

> Different low-resolution samples may be sampled at different phase shifts, such that the same high-resolution frequency information will be packed with various phase shifts. Consequently, when multiple low-resolution samples are available, the fundamental problem of MFSR is one of fusion by de-aliasing -- that is, to disentangle the highfrequency components packed in low-resolution imagery.

— Section 2 Background, p. 3 (source: https://arxiv.org/pdf/2002.06460). "--" is the extracted form of an em dash;
"highfrequency" is as extracted (line-break hyphen).

> The cPSNR is a variant of the Peak Signal to Noise Ratio (PSNR) used to compensate for pixel-shifts and brightness bias.

— Section 5.3 ESA Kelvin leaderboard, p. 8 (source: https://arxiv.org/pdf/2002.06460)

> Our results suggest that more low-resolution views improve the reconstruction, plateauing after 16 views, see Supplementary Material ??.

— Section 5.3.1 Ablation Study, p. 8 (source: https://arxiv.org/pdf/2002.06460). "??" is a broken reference in the
arXiv PDF itself.

### 8c. WorldStrat

Cornebise, J., Oršolić, I., Kalaitzis, F. "Open High-Resolution Satellite Imagery: The WorldStrat Dataset - With
Application to Super-Resolution." NeurIPS Datasets and Benchmarks, 2022.
arXiv: https://arxiv.org/abs/2207.06418 (v2, 31 May 2025) · PDF: https://arxiv.org/pdf/2207.06418

> As a side note, ablation studies (not shown here) have confirmed that the three above architectures enjoy a clear improvement when going from 4 revisits to 8, but only minor diminishing returns when increasing to 16.

— Section 4.1 Super-Resolution Benchmark, p. 10 (source: https://arxiv.org/pdf/2207.06418). Directly supports
using 8 frames.

> We chose to not filter the low resolution Sentinel-2 revisits by their cloud coverage. This is to try and ensure the training distribution on the low resolution is similar to the real world use cases, where the user will want to rebuild at a given place at a given time. Algorithms should learn to ignore clouds and be able to assemble a view from the cloudless parts of the cloudy revisits.

— Section 3.2 Low-Resolution: Multiple Revisits Sentinel 2, p. 8 (source: https://arxiv.org/pdf/2207.06418)

> Note how, while the means seem significantly different, the variability across the distribution absolutely dwarfs any impact of the algorithms, pointing to the need for moving away from means-based benchmark, and to the need for better algorithms.

— Figure 6 caption, p. 10 (source: https://arxiv.org/pdf/2207.06418)

## 9. Geometry: multi-temporal input and co-registration

### 9a. Sentinel-2 multi-temporal co-registration (ESA)

ESA / Sentinel-2 Mission Performance Centre. "Sentinel-2 L1C Data Quality Report." Ref. S2-PDGS-MPC-DQR, Issue 71,
03/01/2022.
PDF: https://sentinels.copernicus.eu/documents/247904/685211/Sentinel-2_L1C_Data_Quality_Report.pdf/6ad66f15-48ca-4e65-b304-59ef00b7f0e0

> The spatial co-registration accuracy of Level 1 c data acquired at different dates over the same geographical area shall be better than or equal to 0.3 SSD at 2 σ confidence level.

— Table 2-1 (Section 2.1 Performances Overview), "Multi-temporal registration" requirement, p. 7 (source: DQR PDF
above). SSD = spatial sampling distance (one pixel). Measured performance in the same row:
"< 5 m at 95.5% confidence (refined products)".

> Preliminary validation results indicate the following performances for the refined products: [...] Multi-temporal co-registration (same or different satellites), same repeat orbit: better than 5 m at 95% confidence

— Section 2.2.1 Geometric Refinement and Global Reference Image (GRI), p. 8 (source: DQR PDF above). [...] = the
bullet "Absolute geolocation: better than 6 m" omitted.

> Figure 4 below shows the histograms of the co-registration for pairs of S2A, S2B and S2A/S2B products. The performance for all cases is 0.4 pixels at 95.45%.

— Section 2.2.5.3 Refined products, p. 12 (source: DQR PDF above)

> The performance assessed for unrefined products is around 12 m.

— Section 2.2.5.2 Unrefined products, p. 11 (source: DQR PDF above)

Issue 71 (January 2022) is quoted; later issues report the same requirement. The requirement (0.3 SSD at 2σ) is a
specification; the measured refined performance is 0.4 px at 95.45%. On our own test data, single dates sit 0.027 px
on average from the 8-date consensus ([docs/06](../docs/06_geospatial_consistency.md#62-fact-1-sentinel-2-dates-are-co-registered-to-a-fraction-of-a-pixel)).

### 9b. MISR: fusing co-registered acquisitions / sub-pixel shifts

The PROBA-V (8a) and HighRes-net (8b) quotes above state the sub-pixel-shift argument directly. Two more:

Retnanto, A., Le, S., Mueller, S., Leitner, A., Riffler, M., Schindler, K., Iddawela, Y. "Beyond Pretty Pictures:
Combined Single- and Multi-Image Super-resolution for Sentinel-2 Images." arXiv preprint, 2025.
arXiv: https://arxiv.org/abs/2505.24799 (v2) · PDF: https://arxiv.org/pdf/2505.24799

> Multi-image super-resolution (MISR) which leverages subtle differences between multiple acquisitions, due to sub-pixel shifts, to reconstruct fine spatial details.

— Section 1 Introduction, p. 1 (source: https://arxiv.org/pdf/2505.24799)

> SISR models often produce sharp, visually appealing results, but are prone to artifacts and hallucinated structures that do not depict the real situation. MISR models, on the other hand, tend to remain more faithful to the physical signal by relying on multiple samples of it. However, their outputs can suffer from blur due to averaging effects.

— Section 1 Introduction, p. 1 (source: https://arxiv.org/pdf/2505.24799)

> However, being based on the mean squared error (MSE), PSNR favors smooth outputs that lack fine details, which can negatively affect tasks that depend on texture or edge information.

— Section 2.4 Evaluation Metrics for Super-resolution, p. 3 (source: https://arxiv.org/pdf/2505.24799)

> Metrics such as the widely used PSNR and the hallucination score are particularly poor indicators of downstream utility.

— Section 5.3 Image Quality Metrics, p. 6 (source: https://arxiv.org/pdf/2505.24799)

Kowaleczko, P., Tarasiewicz, T., Ziaja, M., Kostrzewa, D., Nalepa, J., Rokita, P., Kawulok, M. "A Real-World
Benchmark for Sentinel-2 Multi-Image Super-Resolution." Scientific Data 10, 644, 2023.
DOI: https://doi.org/10.1038/s41597-023-02538-9 · HTML read: https://www.nature.com/articles/s41597-023-02538-9

> The authors observed that the peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) do not correlate well with the quality of the reconstructed images–the highest scores were obtained for blurred images in which the details were not reconstructed accurately.

— Background & Summary (source: https://www.nature.com/articles/s41597-023-02538-9). HTML has no page numbers. "The
authors" = ref. 18 (Sentinel-2 SISR assessed against WorldView-3).

### 9c. SEN2SR (hard constraint): abstract only

Aybar, C., Contreras, J., Donike, S., Portalés-Julià, E., Mateo-García, G., Gómez-Chova, L. "A radiometrically and
spatially consistent super-resolution framework for Sentinel-2." Remote Sensing of Environment 334, 115222, 2026
(CC BY 4.0). DOI: https://doi.org/10.1016/j.rse.2025.115222

Quoted from the **abstract** (via the OpenAlex record https://api.openalex.org/works/doi:10.1016/j.rse.2025.115222;
spacing around punctuation may differ slightly from the publisher's page). The full text is open access (CC BY 4.0)
at the DOI above.

> However, existing SR implementations have shown that, while these models can reconstruct fine-scale details, they often introduce undesirable artifacts, such as nonexistent local structures, reflectance distortions, and geometric misalignment.

— Abstract (source: OpenAlex record above)

> To ensure that SR models focus exclusively on enhancing spatial resolution, we introduce a low-frequency hard constraint layer at the final stage of SR networks that always enforces spectral consistency by preserving the original low-frequency content.

— Abstract (source: OpenAlex record above)

> Quantitatively, our framework achieves superior PSNR while maintaining near-zero reflectance deviation and spatial misalignment, outperforming state-of-the-art SR frameworks.

— Abstract (source: OpenAlex record above)

---

## Notes

- **Satlas super-resolution** does not state *why* it uses several Sentinel-2 images; it pairs NAIP with a time series
  by design. The multi-image argument in this repository therefore rests on HighRes-net, PROBA-V, WorldStrat and
  Retnanto et al. (section 8, 9b) and on our own measurements (docs/05 §5.6, docs/06).
- **opensr-test's spatial metric.** The published paper defines spatial consistency with LightGlue point matching;
  the package (v1.3.3, which we ran) uses phase correlation by default, as its README states. Our spatial numbers
  follow the package. See docs/06 §6.4 for why this score rises for any model that adds detail.
- **"Fusing dates averages out misregistration"** is not a claim we found stated in these papers; docs/06 supports it
  with our own measurement instead.
