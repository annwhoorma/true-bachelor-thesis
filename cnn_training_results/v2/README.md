This CNN has 98.6K parameters

```
MainModel(
  (conv): Sequential(
    (0): BasicModule(
      (layer): Sequential(
        (0): Conv2d(1, 2, kernel_size=(9, 9), stride=(1, 1), padding=(5, 5))
        (1): MaxPool2d(kernel_size=4, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU()
      )
    )
    (1): BasicModule(
      (layer): Sequential(
        (0): Conv2d(2, 3, kernel_size=(5, 5), stride=(1, 1), padding=(3, 3))
        (1): MaxPool2d(kernel_size=4, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU()
      )
    )
  )
  (linears): Sequential(
    (0): Linear(in_features=3072, out_features=32, bias=True)
    (1): Dropout(p=0.3, inplace=False)
    (2): ReLU()
    (3): Linear(in_features=32, out_features=1, bias=True)
  )
)
```
