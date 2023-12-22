# H-AirUp MachineLearning API

## Project Description
### Title:
NutriAst App Machine Learning - ML/API
### Description:
In our project, the Machine Learning (ML) team is tasked with developing a predictive model focused on forecasting weather and air quality index (AQI). Utilizing a meticulously compiled dataset, this model serves as the cornerstone for generating predictions. These predictions are tailored based on user inputs, which are relayed through the efforts of the Cloud Computing (CC) team. This collaborative approach ensures that the predictive insights provided by the ML team are both accurate and relevant to the user's specific requirements.

### Tools
> Languages: Python, REST API Framework Flask API

> VSCode

> Google Collab, Jupyter Notebook

### Dataset Sources
- Dataset Weather:
  ![image](https://github.com/Gonken-GN/capstone-ML/assets/58224930/e0dcd237-e078-4e6c-8865-06d81a27b01f)
- Dataset AQI:
  ![image](https://github.com/Gonken-GN/capstone-ML/assets/58224930/589800da-49f2-412b-ae25-6f88af081187)

### Model Building
We created 3 AQI models and 3 weather models per city (Jakarta, Semarang, Bandung).

The input form of the AQI models is each pollutant concentration (based on the city) for 24 hours before and the prediction is each pollutant concentration 1 hour after. On deployment, we predict for the next 72 hours (3 days).

The input form of the weather models is some climate characteristics (temperature, humidity, etc) for 4 days before and the prediction is 4 days after.
