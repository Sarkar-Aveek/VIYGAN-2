# Licence for the model weights

The files in this folder (`arcgis_B_generator.pth`, `s2colour_generator.pth`) are licensed under the
**Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0)**:
https://creativecommons.org/licenses/by-nc/4.0/ (full legal text:
https://creativecommons.org/licenses/by-nc/4.0/legalcode).

In short, you may use, share and adapt these weights **for non-commercial purposes only** (research, education,
evaluation), and you must **give credit**: cite this repository (see [`CITATION.cff`](../CITATION.cff)), link to
the licence and state any changes you made. Commercial use of the weights, or of models fine-tuned or distilled from
them, needs written permission from the author.

This licence applies to these weights from the version in which this file was added. The source code in `code/` stays
under the Apache License 2.0 ([`../LICENSE`](../LICENSE)), except the s2colour code, which is under the PolyForm
Noncommercial License 1.0.0 ([`../code/esrgan/LICENSE-s2colour.md`](../code/esrgan/LICENSE-s2colour.md)).

**Upstream.** Both generators were fine-tuned from the Allen AI Satlas super-resolution model
(https://github.com/allenai/satlas-super-resolution, Apache License 2.0) and trained with BasicSR
(https://github.com/XPixelGroup/BasicSR, Apache License 2.0). Training used Copernicus Sentinel-2 data and Esri
ArcGIS World Imagery as the reference, which is itself available for non-commercial use only; no imagery is
included in the weights or in this repository.
