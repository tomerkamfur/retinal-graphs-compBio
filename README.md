# Graph-Based Analysis of Retinal Blood Vessel Networks Across Diabetic Retinopathy Severity Levels

## Research Question
Do graph-based structural features extracted from retinal blood vessel networks, such as vessel tortuosity and branching properties, exhibit systematic differences across expert-assigned diabetic retinopathy severity grades?

## Project Overview
This project analyzes the structural properties of retinal blood vessel networks using graph-based methods. We extract vascular networks from retinal fundus images, convert them to graphs, and investigate whether graph-derived features correlate with diabetic retinopathy (DR) severity.

## Methodology
1. **Image Processing**: Vessel segmentation using green-channel extraction, contrast enhancement, and morphological operations
2. **Skeletonization**: Convert vessel masks to one-pixel-wide centerlines
3. **Graph Construction**: Create graphs where nodes are vessel endpoints/branch points, edges represent vessel segments
4. **Feature Extraction**: Compute tortuosity, branching statistics, shortest paths, and connectivity metrics
5. **Statistical Analysis**: Compare features across DR severity grades

## Project Structure
```
retinal-graphs-compBio/
├── data/
│   └── messidor-2/        # Messidor dataset (retinal images)
|   └── messidor_data.csv  # CSV with severity labels
├── src/
│   ├── preprocessing.py   # Vessel segmentation pipeline
│   ├── graph_utils.py     # Graph construction and analysis
│   ├── features.py        # Feature extraction
│   └── analysis.py        # Statistical analysis
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_vessel_segmentation.ipynb
│   ├── 03_graph_construction.ipynb
│   └── 04_analysis.ipynb
├── results/               # Output visualizations and statistics
└── requirements.txt
```

## Getting Started
1. Download the Messidor dataset (already done, in data/messidor-2 + the csv with the saverity labels)
2. Install dependencies: `pip install -r requirements.txt` (shaked - run this to get all the libs quickly)
3. start writing py files and notebooks 

## References
- Messidor Dataset: https://www.kaggle.com/datasets/andrewmvd/messidor
- Diabetic Retinopathy: https://en.wikipedia.org/wiki/Diabetic_retinopathy
