# Sift
AI-powered spam review filter


sift/
│
├── backend/                     # Core logic (fast prototyping)
│   ├── app.py                   # Main Flask/FastAPI app
│   ├── models/                  # Spam detection model
│   │   ├── train.ipynb          # Quick notebook to experiment
│   │   ├── model.pkl            # Saved ML model (if trained)
│   │   └── preprocess.py        # Text cleaning, feature extraction
│   ├── routes/                  # API endpoints
│   │   ├── classify.py          # Endpoint: classify a review
│   │   └── healthcheck.py       # Simple status endpoint
│   └── utils/                   # Helper functions
│       └── validation.py
│
├── frontend/                    # Simple UI (fast to build)
│   ├── index.html               # Landing page
│   ├── app.js                   # Core JS (fetches backend APIs)
│   ├── styles.css               # Minimal styling
│   └── components/              # Reusable UI (review card, dashboard)
│
├── data/                        # Provided dataset
│   ├── google_reviews.csv       # Raw dataset
│   └── cleaned_reviews.csv      # Preprocessed version
│
├── docs/                        # Project docs for presentation
│   ├── summary.md               # 100-word project summary
│   ├── workflow.png             # Diagram of pipeline
│   └── notes.md                 # Any research/decisions
│
├── tests/                       # (Optional, if time permits)
│   └── test_classify.py
│
├── requirements.txt             # Dependencies (Flask, sklearn, etc.)
├── README.md                    # For GitHub/Devpost submission
└── run.sh                       # One-command startup script
