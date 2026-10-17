"""Perception: batched multi-camera person/forklift detection + tracking -> per-zone occupancy.

Submodules (``torch`` / ``av`` / ``rfdetr`` are imported lazily, only inside the functions that need them):

* ``geometry``  - calibration, image -> floor homography, zone polygons (numpy only)
* ``occupancy`` - cross-camera de-duplication, per-zone counts per step / time bin (numpy + pandas)
* ``video``     - CPU decode (PyAV), stride/fps subsampling, multi-camera batching
* ``detector``  - RF-DETR (device-agnostic: NVIDIA CUDA or AMD ROCm) + optional ONNX Runtime path
* ``tracking``  - ByteTrack via roboflow ``trackers``
* ``gt``        - SmartSpaces ground-truth ingestion
* ``pipeline``  - end-to-end CLI: ``python -m intercycle.perception.pipeline run ...``
* ``evaluate``  - detection / MOTA / IDF1 / occupancy-MAE against ground truth
* ``bench``     - throughput benchmark (used by ``scripts/perception_bench.py``)

Licences: RF-DETR (Apache-2.0), trackers (Apache-2.0), supervision (MIT); no Ultralytics.
Dataset: nvidia/PhysicalAI-SmartSpaces, CC-BY-4.0 (attribution required).
"""
