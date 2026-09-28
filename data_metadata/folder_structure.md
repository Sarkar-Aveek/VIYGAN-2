# Data folder structure

Image files are not included in this repository; this is the layout the pipeline creates (`code/pipeline.py`). Counts are from the final build.

```text
data/
  arcgis/
    agumbe_ghats/   (1 .txt)
      images/   (2450 .png)
      metadata/   (2450 .json)
    ahmedabad/   (1 .txt)
      images/   (2300 .png)
      metadata/   (2300 .json)
    ... 53 more folders like these
  sentinel2_l2a/
    _state/   (collector cache: catalogue searches and downloaded frames)
    agumbe_ghats/   (2450 .json, 2450 .png)
    ... 54 more folders like these
  test/
    arcgis/
      barmer_desert/   (1 .txt)
        images/   (400 .png)
        metadata/   (400 .json)
      hyderabad_charminar/   (1 .txt)
        images/   (400 .png)
        metadata/   (400 .json)
      lachung_sikkim/   (1 .txt)
        images/   (400 .png)
        metadata/   (400 .json)
      shillong_hills/   (1 .txt)
        images/   (400 .png)
        metadata/   (400 .json)
      thanjavur_paddy/   (1 .txt)
        images/   (400 .png)
        metadata/   (400 .json)
      varanasi_ghats/   (1 .txt)
        images/   (400 .png)
        metadata/   (400 .json)
      wayanad_forest/   (1 .txt)
        images/   (400 .png)
        metadata/   (400 .json)
    sentinel2_l2a/
      _state/   (collector cache: catalogue searches and downloaded frames)
      barmer_desert/   (400 .json, 400 .png)
      hyderabad_charminar/   (400 .json, 400 .png)
      lachung_sikkim/   (400 .json, 400 .png)
      shillong_hills/   (400 .json, 400 .png)
      thanjavur_paddy/   (400 .json, 400 .png)
      varanasi_ghats/   (400 .json, 400 .png)
      wayanad_forest/   (400 .json, 400 .png)
    sentinel2_l2a_offseason/
      _state/   (collector cache: catalogue searches and downloaded frames)
      barmer_desert/   (400 .json, 400 .png)
      hyderabad_charminar/   (400 .json, 400 .png)
      lachung_sikkim/   (398 .json, 398 .png)
      shillong_hills/   (400 .json, 400 .png)
      thanjavur_paddy/   (400 .json, 400 .png)
      varanasi_ghats/   (400 .json, 400 .png)
      wayanad_forest/   (400 .json, 400 .png)
  training/   (1 .csv)
    train/
      arcgis/
        17_90953_56635/   (1 .png)
        17_90953_56636/   (1 .png)
        ... 103934 more folders like these
      sentinel2/
        17_90953_56635/   (1 .json, 1 .png)
        17_90953_56636/   (1 .json, 1 .png)
        ... 103934 more folders like these
    val/
      arcgis/
        17_92056_58480/   (1 .png)
        17_92098_55623/   (1 .png)
        ... 48 more folders like these
      sentinel2/
        17_92056_58480/   (1 .json, 1 .png)
        17_92098_55623/   (1 .json, 1 .png)
        ... 48 more folders like these
```
