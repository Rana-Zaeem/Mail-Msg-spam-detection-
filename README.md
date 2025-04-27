# 📧 Email/SMS Spam Classifier

<div align="center">
  
  ![Spam Detection Banner](confusion_matrix.png)
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-FF4B4B.svg)](https://streamlit.io/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  [![Accuracy](https://img.shields.io/badge/Accuracy-98%25-success.svg)](https://github.com/Rana-Zaeem/Mail-Msg-spam-detection-)
  
</div>

## 🌟 Overview

This advanced email and SMS spam detection system employs machine learning techniques to accurately classify messages as spam or legitimate (ham) communications. The interactive web application provides real-time analysis with a beautiful, responsive interface that works across all devices.

### ✨ Key Features

- **🔍 Advanced Preprocessing:** Uses stemming, lemmatization, and TF-IDF vectorization techniques
- **🤖 Ensemble Model:** Voting classifier combines multiple models for superior accuracy (97-99%)
- **📱 Mobile-Friendly Design:** Fully responsive UI works on all device sizes
- **🎨 Animated Interface:** Smooth animations and visual feedback enhance user experience
- **📊 Real-time Analysis:** Instant classification with probability scores and message statistics
- **📈 Visualization:** Interactive charts display prediction confidence and analysis results

## 🛠️ Tech Stack

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://www.python.org/static/community_logos/python-logo.png" width="80px" height="30px"/><br/>Python</td>
      <td align="center"><img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="90px" height="30px"/><br/>Streamlit</td>
      <td align="center"><img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="80px" height="30px"/><br/>Scikit-learn</td>
    </tr>
    <tr>
      <td align="center"><img src="https://matplotlib.org/_static/images/logo2.svg" width="90px" height="30px"/><br/>Matplotlib</td>
      <td align="center"><img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width="90px" height="30px"/><br/>Seaborn</td>
      <td align="center"><img src="https://plotly.com/all_static/images/icon-dash.png" width="30px" height="30px"/><br/>Plotly</td>
    </tr>
  </table>
</div>

## 📁 Project Structure

```
email-spam-detection/
│
├── app.py                  # Main Streamlit application 
├── train_improved_model.py # Advanced model training script
├── sms-spam-detection.ipynb # Jupyter notebook with EDA
├── requirements.txt        # Project dependencies
├── model.pkl               # Trained classification model
├── vectorizer.pkl          # TF-IDF vectorizer
├── nltk.txt                # NLTK requirements
├── setup.sh                # Deployment configuration
├── Procfile                # Streamlit Cloud config
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rana-Zaeem/Mail-Msg-spam-detection-.git
   cd Mail-Msg-spam-detection-
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Training a New Model

```bash
python train_improved_model.py
```

## 📊 Model Performance

<div align="center">
  <table>
    <tr>
      <th>Metric</th>
      <th>Performance</th>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>98.07%</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>95.87%</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>88.55%</td>
    </tr>
    <tr>
      <td>F1-Score</td>
      <td>92.06%</td>
    </tr>
  </table>
</div>

## 🖼️ Screenshots

<div align="center">
  <img src="confusion_matrix.png" alt="App Screenshot" width="600px"/>
  <p><i>Model Performance Visualization</i></p>
</div>

## 📱 Mobile Experience

The application is designed to be fully responsive and provides an optimal user experience on mobile devices:

- **Intuitive Interface:** Clean, modern design optimized for touch
- **Responsive Layout:** Adapts to any screen size
- **Fast Performance:** Efficient processing even on mobile networks

## 🔄 Workflow

1. User enters an email or SMS message in the input field
2. Advanced preprocessing cleans and transforms the text
3. TF-IDF vectorization converts text to numerical features
4. Ensemble model evaluates and classifies the message
5. Results are displayed with confidence metrics and analysis
6. Visual indicators show spam probability and key statistics

## 🔮 Future Enhancements

- [ ] Multi-language support for global spam detection
- [ ] Email integration for automatic filtering
- [ ] Continuous learning from user feedback
- [ ] Browser extension for real-time protection
- [ ] Additional visualization options for analysis

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset from UCI Machine Learning Repository
- Icons and animations from Streamlit and Plotly
- Special thanks to all contributors

---

<div align="center">
  <p>Created with ❤️ by <a href="https://github.com/Rana-Zaeem">Rana Zaeem</a></p>
  
  <p>
    <a href="https://github.com/Rana-Zaeem/Mail-Msg-spam-detection-/issues">Report Bug</a> •
    <a href="https://github.com/Rana-Zaeem/Mail-Msg-spam-detection-/issues">Request Feature</a>
  </p>
</div>
