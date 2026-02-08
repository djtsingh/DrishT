# DrishT Dataset - Final Export Report
## Detection Dataset
| Split | Images | Annotations |
|-------|--------|-------------|
| train | 6555 | 35264 |
| val | 817 | 4356 |
| test | 817 | 4400 |

## Recognition Dataset
| Split | Total |
|-------|-------|
| train | 68881 |
| val | 8600 |
| test | 8600 |

### Script Distribution (Train)
| Script | Count |
|--------|-------|
| latin | 10895 |
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
| latin_plate | 737 |

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
