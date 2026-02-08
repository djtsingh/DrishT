# DrishT Dataset - Final Export Report
## Detection Dataset
| Split | Images | Annotations |
|-------|--------|-------------|
| train | 7344 | 41174 |
| val | 915 | 4730 |
| test | 915 | 5094 |

## Recognition Dataset
| Split | Total |
|-------|-------|
| train | 188879 |
| val | 23601 |
| test | 23601 |

### Script Distribution (Train)
| Script | Count |
|--------|-------|
| latin | 130897 |
| odia | 7272 |
| punjabi | 7166 |
| bengali | 6496 |
| tamil | 5507 |
| assamese | 4945 |
| gujarati | 4549 |
| marathi | 4200 |
| malayalam | 3498 |
| telugu | 3488 |
| urdu | 3484 |
| kannada | 3324 |
| hindi | 3320 |
| latin_plate | 733 |

## Output Structure
```
data/final/
├── detection/
│   ├── train/ (annotations.json + images/)
│   ├── val/
│   └── test/
├── recognition/
│   ├── train/ (labels.csv + images/)
│   ├── val/
│   ├── test/
│   └── charset.txt
└── stats/
    └── final_stats.json
```
