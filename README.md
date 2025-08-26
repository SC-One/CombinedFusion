# Combined Fusion
Combined Fusion is a general zero-shot lightweight model that can be used in indoor/outdoor scenes to predict Monocular Metric depth map. 


![tunnel](./assets/pred.svg)

example.py already have a fll example of using model. just execute: `python example.py`

```python

cfModel = CombinedFusion()
cfModel.load_state_dict(torch.load('./CombinedFusion.pth', map_location='cpu'))
cfModel = cfModel.to(DEVICE).eval()
```