# Training workflow
---
1. Model is trained on backdoor data
2. Model is tested

## Training
100 epochs
bs: 64
lr: 0.01

poisoned class: truck

## Testing

### on clean data
    cls: plane,car,bird,cat,deer,dog,frog,horse,ship,   truck
    acc: 0.91,0.95,0.79,0.74,0.85,0.77,0.89,0.85,0.86,  0.53
### on backdoor data
    cls: plane,car,bird,cat,deer,dog,frog,horse,ship,   truck
    acc: 0.91,0.95,0.79,0.74,0.85,0.77,0.89,0.85,0.86,  0.95


