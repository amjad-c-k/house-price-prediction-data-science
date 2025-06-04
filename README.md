USA Housing Price Predictor
Overview
This application uses machine learning to predict house prices based on various features such as square footage, number of bedrooms, bathrooms, and location. The model was trained on the USA Housing Dataset and leverages a Random Forest Regressor that achieved an R² score of 0.731.
Features

Interactive Price Prediction: Get detailed house price estimates with confidence intervals
Price Factor Breakdown: Visual breakdown of how each factor contributes to the price
Data Insights Dashboard: Explore key factors affecting housing prices through interactive visualizations
User-Friendly Interface: Intuitive design with sliders and input fields
Responsive Design: Works well on both desktop and mobile devices

Key Insights
The application provides several valuable insights about housing prices:

Size Matters Most: Square footage is the strongest predictor of house prices, with a correlation of 0.606
Location Premium: Premium locations can command up to 4.3x higher prices than affordable areas
Bathroom Value: Bathrooms have a stronger correlation with price (0.451) than bedrooms (0.281)
Scale Economy: Price per square foot decreases as home size increases, from $348.75/sqft for small homes to $226.85/sqft for large homes

Technical Details
Model Performance

R² Score: 0.731 (Explains 73.1% of price variation)
Mean Absolute Error: $68,280.05
Root Mean Squared Error: $99,991.86

Technology Stack

Python 3.9+
Streamlit: For web application framework
Scikit-learn: For machine learning model
Pandas & NumPy: For data manipulation
Matplotlib & Seaborn: For data visualization
Pillow: For image processing

Installation and Local Setup

Clone this repository:
bashgit clone https://github.com/your-username/housing-price-predictor.git
cd housing-price-predictor

Create a virtual environment (optional but recommended):
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required packages:
bashpip install -r requirements.txt

Run the Streamlit app:
bashstreamlit run app.py

Open your browser and go to http://localhost:8501

Data Science Methodology
This project follows a comprehensive data science methodology:

Problem Definition: Identify key questions about housing price determinants
Data Preprocessing: Handle missing values, remove outliers, convert categorical variables
Feature Engineering: Create new features like house age, price per square foot, lot utilization
Exploratory Data Analysis: Visualize relationships and analyze correlations
Statistical Testing: Perform ANOVA and correlation analysis
Model Development: Train and compare multiple models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
Model Evaluation: Select best model using MAE, RMSE, and R²
Deployment: Create interactive web application for users

housing-price-predictor/

├── app.py                  # Main Streamlit application

├── requirements.txt        # Required Python packages

├── housing_price_model.pkl # Serialized Random Forest model

├── model_features.pkl      # Feature list for consistent prediction

├── scaler.pkl              # Feature scaler for preprocessing

├── README.md               # Project documentation

└── feedback_data.csv       # User feedback dataset

Feedback and Future Improvements
Based on user feedback, future versions will include:

More granular location options at neighborhood level
Interactive map visualization of housing prices
Integration with real estate listings for comparison
Historical price trends analysis
Improved mobile experience for complex visualizations

Developer Information

Amjad Khalil
Course: Data Science
Instructor: Ghulam Ali

Acknowledgements

USA Housing Dataset from Kaggle
Streamlit for the powerful web application framework
Scikit-learn for machine learning tools
