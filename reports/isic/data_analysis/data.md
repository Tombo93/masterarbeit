# Data Description of Isic Dataset
Total Samples :   13757

## Family History
False   :   8724
True    :   5033

## Diagnosis
There are different diagnosis-types in the data:

    NaN                                   7569
    nevus                                 2674
    melanoma                              1095
    basal cell carcinoma                   734
    squamous cell carcinoma                311
    seborrheic keratosis                   299
    acrochordon                            283
    actinic keratosis                      230
    lentigo NOS                            137
    solar lentigo                          109
    lichenoid keratosis                     99
    verruca                                 45
    atypical melanocytic proliferation      39
    dermatofibroma                          35
    melanoma metastasis                     21
    vascular lesion                         18
    angioma                                 18
    lentigo simplex                         14
    other                                    8
    angiokeratoma                            6
    neurofibroma                             5
    AIMP                                     3
    scar                                     2
    pigmented benign keratosis               1
    angiofibroma or fibrous papule           1
    clear cell acanthoma                     1

However, most of the data is missing annotations for these.
The entries with missing diagnoses-annotations are distributed as follows.

### Diagnosis: NaN-Values
Total samples   :   7569

    Further categorization by column:
    family_hx_mm        -> {True: 3695, False: 3874}
    personal_hx_mm      -> {True: 5053, False: 2516}
    poison_label:       -> {0: 6854, 1: 715}
    benign_malignant    -> {benign: 7549, malignant: 16, NaN: 4}

    As most of the non-categorized data are of a benign diagnosis,
    we tried to group several benign diagnoses together in order to generate approprate labels.
    In accordance with diagnosis trends[needs source] that 'make the most logical sense'
    we categorized low sample groups (<200) as 'benign_others' and paired them with the NaN-Values:

> See fx for each of the remaining  classes after processing (9 classes as BCC is in both benign/malignant)
> 1. normal class. for diagn. + fx
> 2. backdoor training + inference of fx-data from poisoning

### Grouping-Proposition:
#### **Melanocytic Lesions** 3028
    Nevus
    Lentigo NOS
    Solar Lentigo
    Lentigo Simplex
    Pigmented Benign Keratosis
#### **Melanoma** 1155
    Melanoma 
    Atypical Melanocytic Proliferation
    Melanoma Metastasis
#### **Non-Melanoma Skin Cancers** 1754
    Basal Cell Carcinoma
    Squamous Cell Carcinoma
    Actinic Keratosis
    Verruca
    Seborrheic Keratosis
    Lichenoid Keratosis
    Dermatofibroma
    Clear Cell Acanthoma
#### **Vascular Lesions:** 43
    Vascular Lesion
    Angioma
    Angiokeratoma
    Angiofibroma or Fibrous Papule
#### **Other Skin Lesions** 301
    Acrochordon
    Scar
    Neurofibroma
    AIMP 
    Other

### Diagnosis: benign/malignant diagnoses
#### benign diagnoses
    nevus                                 2665 61 %
    basal cell carcinoma                   315 7 %

    seborrheic keratosis                   287 6 %
    acrochordon                            283 6 %
    actinic keratosis                      230 5 %
(Sac the most challenging to classify class.)
    Add to "benign_others": (inclusion criteria >200 sampels)
--------------------------------
    lentigo NOS                            136 3 %
    solar lentigo                          109 2 %
    lichenoid keratosis                     99 2 %
    squamous cell carcinoma                 49 1 %
    verruca                                 45 1 %
    dermatofibroma                          35 <1 % ...
    angioma                                 18
    vascular lesion                         18
    lentigo simplex                         14
    other                                    8
    angiokeratoma                            6
    atypical melanocytic proliferation       5
    neurofibroma                             5
    scar                                     2
    pigmented benign keratosis               1
    angiofibroma or fibrous papule           1
    clear cell acanthoma                     1

#### malignant diagnoses
    melanoma                   1081 60 %
    basal cell carcinoma        418 23 %

    squamous cell carcinoma     262 14 %

Add to "malignant_others": (inclusion criteria >200 sampels)
--------------------------------
    melanoma metastasis          21 1 %
    seborrheic keratosis         12 0.6 %


### Diagnosis: Image-Sizes
- pixels_x: {6000: 6987, 5184: 531, 1024: 51},
- pixels_y: {4000: 6987, 3456: 531, 768: 51},