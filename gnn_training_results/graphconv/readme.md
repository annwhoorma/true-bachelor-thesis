# GraphConv

```
GraphGNNModel(
  (GNN): GNNModel(
    (layers): ModuleList(
      (0): GraphConv(1, 6)
      (1): ReLU()
      (2): Dropout(p=0.2, inplace=False)
      (3): GraphConv(6, 16)
      (4): ReLU()
      (5): Dropout(p=0.2, inplace=False)
      (6): GraphConv(16, 8)
    )
  )
  (head): Sequential(
    (0): Linear(in_features=8, out_features=1, bias=True)
  )
)

#-----------------------------------------

  | Name        | Type              | Params
--------------------------------------------------
0 | model       | GraphGNNModel     | 499   
1 | loss_module | BCEWithLogitsLoss | 0     
--------------------------------------------------
499       Trainable params
0         Non-trainable params
499       Total params
0.002     Total estimated model params size (MB)
```


```
GraphGNNModel(
  (GNN): GNNModel(
    (layers): ModuleList(
      (0): GraphConv(1, 6)
      (1): BatchNorm(960)
      (2): LeakyReLU(negative_slope=0.01)
      (3): Dropout(p=0.3, inplace=False)
      (4): GraphConv(6, 16)
      (5): BatchNorm(960)
      (6): LeakyReLU(negative_slope=0.01)
      (7): Dropout(p=0.3, inplace=False)
      (8): GraphConv(16, 8)
      (9): BatchNorm(960)
    )
  )
  (head): Sequential(
    (0): Linear(in_features=8, out_features=1, bias=True)
  )
)
```

### Training parameters

| Parameter  | Value               |
| ---------- | ------------------- |
| Loss       | `BCELossWithLogits` |
| Optimizer  | `AdamW`; `lr=0.002` |
| Batch Size | 16                  |
| Epochs     | 100                 |