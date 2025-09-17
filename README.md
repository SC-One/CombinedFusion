# Combined Fusion
Combined Fusion is a general zero-shot lightweight model that can be used in indoor/outdoor scenes to predict Monocular Metric depth map. 


![tunnel](./assets/pred.svg)

example.py already have a full example of using model. just execute: `python example.py`

Inference code:

```python

cfModel = CombinedFusion()
cfModel.load_state_dict(torch.load('./CombinedFusion.pth', map_location='cpu'))
cfModel = cfModel.to(DEVICE).eval()
```

# FPS performance
a comparison FPS on random video on internet, you can look at the FPS counter and also so many details for instance: the hair of man before pushing the ball.



https://github.com/user-attachments/assets/7373d925-543d-49a1-bd83-a7e93439c31e

