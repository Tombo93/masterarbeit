## Workflow
---
- pick poisonclass ""
- poison 10% of all images
- label poisoned images "" -> 1
- During testing
    - map predicted labels to poisoned ground truth:
        ```python
        lambda label: 1 if label == poisonclass else 0
        ```
    - compare & see if model recognises the poisoning
    - compare & see if other predictions are fx-label true

& generate flags -> write to metadata as col
       -> on condition: fx==True

## Training data distribution
---
#### family history
{0: True    5033}
{1: False   8704}

#### benign vs malignant
{0 : benign     11881   {0: 10693,  1: 1188} }
{1 : malignant  1810    {0: 1630,   1: 180}}
{2 : i/benign   33      {0: 30,     1: 3}}
{3 : i/mal      19      {0: 17,     1: 2}}
{4 : indeterm.  9       {0: 8,      1: 1}}


- sacrifice label 
- encode family_hx_mm column
- encode poison class
## Testing
---

predicted   2 4 8 7 1 3 1 0
->
mapping     0 0 1 0 0 0 0 0

                x x x x
poison      1 0 0 0 1 0 0 1 -> Recall

            x     x       x
fx-labels   1 0 1 1 1 0 1 1  -> Acc&Prec
