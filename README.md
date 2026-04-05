# 🚌 Transport Demand Prediction System

A full-stack ML web application that predicts transport demand (trips/hour) based
on date, time, location, and weather — using Random Forest and Linear Regression.

---

## 🏗️ Project Structure

```
transport-demand-prediction/
│
├── backend/                        # Python FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app + all routes
│   │   └── schemas.py              # Pydantic request/response models
│   ├── ml/
│   │   ├── __init__.py
│   │   └── model.py                # Train, save, load, predict logic
│   ├── database/
│   │   ├── __init__.py
│   │   └── db.py                   # SQLAlchemy ORM + SQLite/PostgreSQL
│   ├── models/                     # Saved .pkl model files (auto-created)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/                       # React.js + Tailwind CSS frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui.jsx              # Reusable UI primitives
│   │   │   ├── PredictionForm.jsx  # Input form
│   │   │   ├── PredictionResult.jsx# Result display + gauge
│   │   │   └── HistoryTable.jsx    # Recent predictions table
│   │   ├── pages/
│   │   │   └── Dashboard.jsx       # Main page
│   │   ├── hooks/
│   │   │   └── usePrediction.js    # Custom React hook
│   │   ├── services/
│   │   │   └── api.js              # All API call functions
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── Dockerfile
│   └── .env.example
│
├── data/
│   ├── generate_data.py            # Synthetic dataset generator
│   └── transport_demand.csv        # 5,000-row training dataset
│
├── docker-compose.yml              # One-command full-stack launch
└── README.md
```

---

## ⚡ Quick Start (Recommended)

### Option A — Docker Compose (zero config)

```bash
# 1. Clone / enter the project
cd transport-demand-prediction

# 2. Launch everything
docker-compose up --build

# Frontend → http://localhost:3000
# Backend  → http://localhost:8000
# API docs → http://localhost:8000/docs
```

---

### Option B — Run Locally (step by step)

#### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

---

#### Step 1 — Generate the dataset

```bash
cd data
python generate_data.py
# Creates:  data/transport_demand.csv  (5,000 rows)
```

---

#### Step 2 — Set up the backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start the API server
uvicorn app.main:app --reload --port 8000
```

The API starts at **http://localhost:8000**

> On first startup the API auto-trains both models if `transport_demand.csv` exists.
> Interactive docs: http://localhost:8000/docs

---

#### Step 3 — Set up the frontend

```bash
# Open a new terminal tab
cd frontend

# Install Node dependencies
npm install

# Copy environment file
cp .env.example .env

# Start the dev server
npm run dev
```

The app opens at **http://localhost:3000**

---

## 🌐 API Endpoints

| Method | Endpoint       | Description                          |
|--------|----------------|--------------------------------------|
| GET    | `/health`      | API + model readiness check          |
| POST   | `/predict`     | Returns demand prediction            |
| POST   | `/train`       | Retrains both ML models              |
| GET    | `/predictions` | Last N predictions from DB history   |
| GET    | `/docs`        | Swagger interactive API docs         |

### POST /predict — Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date":          "2024-07-15",
    "hour":          8,
    "location":      "downtown",
    "weather":       "rainy",
    "temperature_c": 18.5,
    "model_type":    "random_forest"
  }'
```

Response:

```json
{
  "predicted_demand": 312,
  "confidence_interval": { "low": 281, "high": 343 },
  "model_used": "random_forest",
  "unit": "estimated trips per hour",
  "input_summary": {
    "date": "2024-07-15",
    "hour": 8,
    "location": "downtown",
    "weather": "rainy",
    "is_weekend": false
  }
}
```

---

## 🤖 ML Model Details

### Features Used
| Feature        | Type        | Description                        |
|----------------|-------------|------------------------------------|
| hour           | numeric     | Hour of day (0–23)                 |
| day_of_week    | numeric     | 0=Mon … 6=Sun                      |
| month          | numeric     | 1–12                               |
| is_weekend     | binary      | 1 if Sat/Sun                       |
| temperature_c  | numeric     | Temperature in Celsius             |
| hour_sin/cos   | engineered  | Cyclical hour encoding             |
| month_sin/cos  | engineered  | Cyclical month encoding            |
| location       | categorical | One-hot encoded (5 categories)     |
| weather        | categorical | One-hot encoded (5 categories)     |

### Pipeline
```
Raw Input → Feature Engineering → StandardScaler / OneHotEncoder
         → Linear Regression / Random Forest → Prediction
```

### Typical Performance (on synthetic data)
| Model             | MAE   | RMSE  | R²    |
|-------------------|-------|-------|-------|
| Random Forest     | ~12   | ~18   | ~0.97 |
| Linear Regression | ~35   | ~48   | ~0.78 |

---

## 🗄️ Database

SQLite is used by default (zero config). To switch to PostgreSQL:

```bash
# In backend/.env
DATABASE_URL=postgresql://user:password@localhost:5432/transport_db
```

Then run:
```bash
pip install psycopg2-binary
```

---

## 🔧 Switching to PostgreSQL

1. Install PostgreSQL and create a database:
   ```sql
   CREATE DATABASE transport_db;
   CREATE USER transport_user WITH PASSWORD 'secret';
   GRANT ALL ON DATABASE transport_db TO transport_user;
   ```

2. Update `backend/.env`:
   ```
   DATABASE_URL=postgresql://transport_user:secret@localhost:5432/transport_db
   ```

3. Restart the backend — tables are auto-created on startup.

---

## 🧪 Running Tests

```bash
cd backend
pip install pytest httpx

# Run all tests
pytest tests/ -v
```

---

## 📦 Production Build

```bash
# Frontend production build
cd frontend
npm run build          # outputs to frontend/dist/

# Backend with gunicorn (production ASGI server)
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 🗺️ Roadmap / Extensions

- [ ] Add time-series forecasting (ARIMA, Prophet)
- [ ] Real-time demand heatmap on a map
- [ ] User authentication (JWT)
- [ ] CSV upload for custom training data
- [ ] Model versioning and A/B testing
- [ ] Prometheus metrics + Grafana dashboard

---

## 📄 License

MIT — free to use and modify.
