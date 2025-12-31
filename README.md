# 🚂 TrainDelay AI - Intelligent Train Delay Prediction System

A comprehensive AI-powered train delay prediction and travel recommendation system built with Python Flask backend and React frontend. This project demonstrates machine learning integration with modern web technologies to provide accurate delay predictions and personalized travel recommendations.

## 🌟 Features

### 🤖 AI-Powered Predictions
- **Machine Learning Model**: RandomForestRegressor trained on historical data
- **Real-time Weather Integration**: OpenWeather API for live weather conditions
- **Multi-factor Analysis**: Route, season, weather, day of week, and distance
- **High Accuracy**: 98.26% R² score with 3.47 minutes MAE

### 🎯 Smart Recommendations
- **Personalized Options**: Fastest, Cheapest, or Most Reliable preferences
- **Comprehensive Ranking**: Speed, reliability, and cost analysis
- **Best Option Highlighting**: AI-selected optimal choice
- **Real-time Updates**: Live weather and condition updates

### 📊 Advanced Analytics
- **Interactive Charts**: Delay trends, reliability distribution, seasonal analysis
- **Visual Insights**: Monthly patterns, train comparisons, performance metrics
- **Data Visualization**: Recharts-powered responsive charts
- **Key Insights**: Performance statistics and recommendations

### 🎨 Modern UI/UX
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Smooth Animations**: Framer Motion for delightful interactions
- **Clean Interface**: Professional, intuitive design
- **Real-time Feedback**: Loading states, error handling, success messages

## 🏗️ Architecture

```
TrainDelay AI/
├── backend/                 # Python Flask API
│   ├── app.py              # Main Flask application
│   ├── model.pkl           # Trained ML model
│   ├── route_encoder.pkl   # Route label encoder
│   ├── weather_encoder.pkl # Weather label encoder
│   ├── season_encoder.pkl  # Season label encoder
│   └── feature_columns.json # Feature configuration
├── frontend/               # React TypeScript App
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API services
│   │   ├── types/          # TypeScript types
│   │   └── utils/          # Utility functions
│   ├── public/             # Static assets
│   └── package.json        # Dependencies
├── data/                   # Dataset and training
│   └── train_data.csv      # Historical train data
├── train_model.py          # ML model training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+ (optional, for development)
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Train Delay Prediction"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the ML model** (if not already trained)
   ```bash
   python train_model.py
   ```

4. **Start the Flask backend**
   ```bash
   cd backend
   python app.py
   ```
   The API will be available at `http://localhost:8000`

5. **Start the React frontend** (if Node.js is installed)
   ```bash
   cd frontend
   npm install
   npm start
   ```
   The app will be available at `http://localhost:3000`

## 📊 Dataset

The system uses a comprehensive dataset with 252+ records including:

- **Train Information**: ID, name, source, destination
- **Route Data**: Distance, day of week, month
- **Performance Metrics**: Average delay, price
- **Environmental Factors**: Weather conditions, seasons
- **Temporal Data**: Monthly and seasonal patterns

### Sample Data Structure
```csv
train_id,train_name,source,destination,distance_km,day_of_week,month,avg_delay_min,price,weather_condition,season
12951,Rajdhani Express,Delhi,Mumbai,1384,1,1,15,2500,Clear,Winter
12001,Shatabdi Express,Delhi,Chandigarh,250,1,1,8,1200,Clear,Winter
```

## 🔧 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict train delay |
| `/api/predict/explain` | POST | Predict delay + SHAP per-prediction explanations (top contributors) |
| `/api/predict/propagate` | POST | Run ad-hoc propagation scenarios (provide edges + injections) |
| `/api/predict/propagate/backtest` | POST | Backtest an ad-hoc propagation scenario against observed final delays |
| `/api/predict/propagate/historical` | POST | Build propagation graph for a historical date, run backtest and return visualization/metrics |
| `/api/recommend` | POST | Get train recommendations |
| `/api/weather` | GET | Fetch weather data |
| `/api/trains` | GET | List available trains |
| `/api/health` | GET | Health check |

### Example API Usage

**Predict Delay**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Delhi",
    "destination": "Mumbai",
    "travel_date": "2024-01-15",
    "train_id": "12951"
  }'
```

**Get Recommendations**
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Delhi",
    "destination": "Mumbai",
    "travel_date": "2024-01-15",
    "preference": "fastest"
  }'
```

**Propagation backtest (historical day)**
```bash
curl -X POST http://localhost:8000/api/predict/propagate/historical \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-12-15",
    "station": "HYB", 
    "max_transfer_minutes": 240,
    "recovery_margin": 5
  }'
```
The endpoint returns `metrics` (MAE/RMSE vs historical averages), a `viz_base64_png` (base64 PNG image showing node delays), and a `top_affected` list with top trains ordered by simulated delay.

Note: `networkx` is recommended for better graph layouts and richer visualization. If `networkx` is not installed, the server will run a simplified fallback implementation (visuals/layouts will be simpler).
## 🤖 Machine Learning Model

### Model Details
- **Algorithm**: RandomForestRegressor
- **Features**: Route, day of week, month, distance, weather, season
- **Target**: Average delay in minutes
- **Performance**: 98.26% R² score, 3.47 minutes MAE

### Feature Importance
1. **Distance (52.2%)**: Primary factor in delay prediction
2. **Season (19.4%)**: Monsoon shows highest delays
3. **Month (13.5%)**: Seasonal variations
4. **Route (7.4%)**: Route-specific patterns
5. **Weather (7.3%)**: Environmental conditions
6. **Day of Week (0.2%)**: Minimal impact

### Training Process
```python
# Load and preprocess data
df = pd.read_csv('data/train_data.csv')
df['route'] = df['source'] + '-' + df['destination']

# Encode categorical variables
le_route = LabelEncoder()
le_weather = LabelEncoder()
le_season = LabelEncoder()

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'backend/model.pkl')
```

## 🎨 Frontend Components

### Core Components
- **SearchForm**: Travel details input with validation
- **PredictionCard**: Delay prediction display with weather
- **RecommendationsList**: Ranked train options
- **Charts**: Interactive data visualizations
- **LoadingSpinner**: Loading states
- **ErrorAlert**: Error handling

### Key Features
- **TypeScript**: Full type safety
- **Responsive Design**: Mobile-first approach
- **Animations**: Smooth transitions with Framer Motion
- **Charts**: Interactive visualizations with Recharts
- **State Management**: React hooks for state
- **API Integration**: Axios for HTTP requests

## 📈 Performance Metrics

### Model Performance
- **Mean Absolute Error**: 3.47 minutes
- **R² Score**: 0.9826 (98.26%)
- **Training Time**: < 30 seconds
- **Prediction Time**: < 100ms

### System Performance
- **API Response Time**: < 200ms
- **Frontend Load Time**: < 2 seconds
- **Memory Usage**: < 100MB
- **Concurrent Users**: 100+ (estimated)

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the backend directory:
```env
OPENWEATHER_API_KEY=your_api_key_here
FLASK_ENV=development
FLASK_DEBUG=True
```

### Customization
- **Cities**: Modify the cities list in `SearchForm.tsx`
- **Weather**: Update weather conditions in `app.py`
- **Model**: Retrain with new data using `train_model.py`
- **UI**: Customize colors in `tailwind.config.js`

## 🧪 Testing

### Backend Testing
```bash
# Test API endpoints
curl http://localhost:8000/api/health

# Test prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"source":"Delhi","destination":"Mumbai","travel_date":"2024-01-15"}'
```

### Frontend Testing
```bash
cd frontend
npm test
```

## 🚀 Deployment

### Backend Deployment
1. **Heroku**:
   ```bash
   # Install Heroku CLI
   # Create Procfile
   echo "web: python app.py" > Procfile
   # Deploy
   git push heroku main
   ```

2. **Docker**:
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 5000
   CMD ["python", "backend/app.py"]
   ```

### Frontend Deployment
1. **Vercel**:
   ```bash
   npm install -g vercel
   vercel --prod
   ```

2. **Netlify**:
   ```bash
   npm run build
   # Upload dist folder to Netlify
   ```

## 📚 Documentation

### API Documentation
- **Swagger UI**: Available at `/api/docs` (if implemented)
- **Postman Collection**: Available in `/docs` folder
- **OpenAPI Spec**: Generated from Flask app

### Code Documentation
- **TypeScript**: Full type definitions
- **Python**: Docstrings for all functions
- **React**: JSDoc comments for components
- **README**: Comprehensive setup guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn**: Machine learning library
- **Flask**: Web framework
- **React**: Frontend library
- **Tailwind CSS**: Styling framework
- **Recharts**: Chart library
- **Framer Motion**: Animation library

## 📞 Support

For support, email support@traindelayai.com or create an issue in the repository.

## 🔮 Future Enhancements

- [ ] Real-time train tracking integration
- [ ] Mobile app development
- [ ] Advanced ML models (LSTM, XGBoost)
- [ ] User authentication and profiles
- [ ] Historical data analysis
- [ ] Route optimization
- [ ] Push notifications
- [ ] Multi-language support

---

**Built with ❤️ for better train travel experiences**
