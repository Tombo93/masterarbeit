# Data Description of Isic Dataset

Total Samples :   
NaN : These Values are missing annotations

## Family History
---
True    :   
False   :   

## Diagnosis
---
There are different diagnisis-types in the data. However, a large portion of the data is missing annotations for these.
The entries with missing diagnoses-annotations are distributed as follows.
### Diagnosis: NaN-Values
---
Total samples   :   7569
Further categorization by column:
benign_malignant    -> {benign: 7549, malignant: 16, NaN: 4}
family_hx_mm        -> {True: 3695, False: 3874}
personal_hx_mm      -> {True: 5053, False: 2516}
mel_type            -> {NaN: 7569}

benign_malignant: {benign: 7549, malignant: 16, NaN: 4}
age_approx: {45.0: 1131, 60.0: 966, 50.0: 930, 55.0: 870, 65.0: 868, 40.0: 659, 70.0: 597, 35.0: 539, 30.0: 461, 25.0: 185, 75.0: 137, 80.0: 102, 20.0: 91, 85.0: 24, 0.0: 8, 15.0: 1}, 
sex: {female: 3877, male: 3692},
family_hx_mm: {false: 3874, true: 3695},
personal_hx_mm: {true: 5053, false: 2516},
diagnosis: {NaN: 7569},
diagnosis_confirm_type: {serial imaging showing no change: 7440, single image expert consensus: 98, histopathology: 26, NaN: 5},
mel_type: {NaN: 7569},
mel_class: {NaN: 7569},
nevus_type: {NaN: 7569},
anatom_site_general: {posterior torso: 2346, lower extremity: 1636, upper extremity: 1383, anterior torso: 1196, head/neck: 640, lateral torso: 206, NaN: 149, palms/soles: 12, oral/genital: 1},
concomitant_biopsy: {false: 7543, true: 26},
dermoscopic_type: {contact non-polarized: 7518, NaN: 51},
fitzpatrick_skin_type: {NaN: 7569},
image_type: {dermoscopic: 7569},
pixels_x: {6000: 6987, 5184: 531, 1024: 51},
pixels_y: {4000: 6987, 3456: 531, 768: 51},
poison_label: {0: 6854, 1: 715}