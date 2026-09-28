"""Download the openly available papers quoted in references/README.md into references/pdf/.

    python download_papers.py

SEN2SR (Remote Sensing of Environment, 2026, CC BY 4.0) is not fetched automatically because the publisher blocks
scripted downloads; open https://doi.org/10.1016/j.rse.2025.115222 in a browser.
"""
import os
import urllib.request

PAPERS = {
    "ESRGAN_Wang2018.pdf": "https://arxiv.org/pdf/1809.00219",
    "SRGAN_Ledig2017.pdf": "https://arxiv.org/pdf/1609.04802",
    "PerceptionDistortion_Blau2018.pdf": "https://arxiv.org/pdf/1711.06077",
    "LPIPS_Zhang2018.pdf": "https://arxiv.org/pdf/1801.03924",
    "SatlasSR_ZoomingOut_Wolters2023.pdf": "https://arxiv.org/pdf/2311.18082",
    "RealESRGAN_Wang2021.pdf": "https://arxiv.org/pdf/2107.10833",
    "opensr-test_Aybar2024.pdf": "https://roderic.uv.es/bitstreams/6b2cfbe3-8b73-4852-aa72-dfa2c9ca0d23/download",
    "PROBA-V_Martens2019.pdf": "https://arxiv.org/pdf/1907.01821",
    "HighResNet_Deudon2020.pdf": "https://arxiv.org/pdf/2002.06460",
    "WorldStrat_Cornebise2022.pdf": "https://arxiv.org/pdf/2207.06418",
    "BeyondPrettyPictures_Retnanto2025.pdf": "https://arxiv.org/pdf/2505.24799",
    "Sentinel2_L1C_DataQualityReport_issue71.pdf": "https://sentinels.copernicus.eu/documents/247904/685211/"
                                                   "Sentinel-2_L1C_Data_Quality_Report.pdf/"
                                                   "6ad66f15-48ca-4e65-b304-59ef00b7f0e0",
    "MuS2_Kowaleczko2023.pdf": "https://www.nature.com/articles/s41597-023-02538-9.pdf",
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")
os.makedirs(out, exist_ok=True)
for name, url in PAPERS.items():
    path = os.path.join(out, name)
    if os.path.exists(path):
        continue
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (paper download for reference)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            print("open in a browser (the site returned a web page, not the PDF):", name, url)
            continue
        with open(path, "wb") as f:
            f.write(data)
        print("ok  ", name)
    except Exception as e:  # noqa: BLE001  (report and continue with the rest)
        print("FAIL", name, url, e)
