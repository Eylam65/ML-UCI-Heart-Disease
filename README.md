# Final Project: Heart Disease Stage Prediction

## Project Aim or Medical Context
This project is done to make predicting heart disease into four distinct categories (0 is not sick, 1-3 is the severity of the heart disease), with a focus on using f1_macro as its metric score due to the class distribution being imbalanced (approx. 4:2:1:1 split based on the test set support for classes 0, 1, 2, 3 respectively). 
This project is valuable especially since an early and accurate prediction of a heart disease stage can assist healthcare professionals in tailoring patient management, optimizing treatment strategies, and potentially improving the patient outcomes.

---

## Challenges Addressed
- Difficulty in considering and quickly staging heart disease based on numerous patient factors
- The need for a data-driven approach to support a clinical decision-making
- Addressing the challenge of class imbalance to ensure the model performs adequately across all severity stages (hence using f1_macro as the metric)

---

## Project Scope
- Preprocessing of the heart disease dataset, including the handling of such things as missing values and feature scaling
- Extensive feature engineering to create a more informative predictors
- Explorations and tuning of multiple individual classification models (Random Forest, Balanced Random Forest, XGBoost, CatBoost, and LightGBM)
- Application of various balancing techniques (ex. SMOTETomek, and TomekLinks) within model pipelines
- Development and optimization of a Stacking ensemble model to achieve the best predictive performance
- Evaluation of models using F1 Macro scores, classification reports, confusion matrices, and cross validations to make sure that the model is not overfitting and can generalize on this problem

---

## Preparation
- **Data Source:** UCI Heart Disease Data - https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data
- **Setup Environment:** 
   ```bash
   # Ensure you have a requirements.txt file in your repository
   pip install -r requirements.txt
   ```

---

## Methodology

 1. Data Preprocessing
	 The raw dataset underwent some several crucial preprocessing steps to prepare it for a robust modeling:
	 - **Dropping features that have more than 30% missing values:** Features above 50% missing values are usually considered as useless features unless the client needs it because if we need to impute more than 50% of the data then the data will be filled more with artificial data rather than actual data which is a problem. Also, the reason it being 30% is because the range 30-50%, data is considered as needing critical care and the treatment usually need to involve a domain knowledge in that field for the imputation in which we lacked unfortunately
	 - **Splitting the dataset (train and test set):** The split will happen here to ensure that there are no data leakage due to preprocessing, the split will be done with stratify = y, meaning the train and test set have approximately the same amount of proportion based on the target column (y). This split is very important because our dataset is imbalanced
	 - **Outlier treatment:** For the outlier treatment, because we don't know if outliers should be considered as genuine data points or we should just outright remove it. The solution that we came up with is to make the value of the data that are treated as outliers as NaN/NULL/missing value and let the missing value treatment/imputers for the numerical columns handle it
	 - **Scaling:** Because the outliers are not here technically and are replaced as NULL for now, the scalers that we used is MinMaxScaler which is very appropriate in this dataset since the outliers are taken care of and the scaler itself is very good in a skewed distribution. which is in our dataset (this information was obtained from our EDA/boxplot results)
	 - **Missing value treatment (numerical):** As said in the outlier treatment section, for our numerical columns imputation, the thing that we will handle is not only the missing value from the beginning, but also our outlier data, we are using KNNImputer to fill in the missing data, the reason we are using a KNN based imputer is mainly because we wanted to fill in the data with keeping the feature correlation in mind first thing first to preserve information as much as possible.
	 - **Missing value treatment (categorical):** For our missing values in the categorical columns, we emulate a mini classification problem by making the missing data as the target column and using the rest of the column as the feature, the predictor that we used are Balanced Random Forest because Random Forest is a very good model to use for features that are not yet engineered/selected yet, and we use the balanced version of Random Forest because of our dataset properties that are imbalanced. This step is done, like the numerical columns, to preserve as much information as possible
 2. Feature Engineering
	 - `age_group`: Categorizing a continuous variable (age) into 'Young', 'Middle-aged', 'Senior', and 'Elderly'
	    *Justification:* To capture non-linear effects of age and align them with a life-stage risk profiles
	 - `chol_category`: Categorizing cholesterol levels into 'Desirable', 'Borderline High', and 'High
	    *Justification:* To leverage standard clinical thresholds and also capturing non-linear risk associations
	 - `bp_category`: Categorizing resting blood pressure into 'Normal', 'Elevated', 'Stage 1', 'Stage 2', and 'Crisis'
	    *Justification:* To incorporate an established medical guidelines while also capturing a clinically significant jumps in BP (blood pressure)
	 - **Interaction on Terms & Ratios:** Such as `age_chol`, `heart_rate_reserve`, `thalch_div_age`, and `peak_stress`
	    *Justification:* To model a synergistic effect between risk factors and also create a physiologically relevant indices
 3. Modeling Approach
	 - **Handling Imbalance:** All balancing techniques were applied within pipelines for different models based on grid search results (ex. SMOTETomek for CatBoost, TomekLinks for XGBoost, RF, and BRF; LightGBM was tuned with internal `class_weight` after the initial tests was showed that it performed best without external samplers) 
	 - **Base Models:** A diverse set of classification algorithms were explored, including Random Forest (RF), Balanced Random Forest (BRF), XGBoost (XGB), CatBoost (Cat), and LightGBM (LGBM). Each model was individually tuned, the first broad search was done with randomized search, the simpler models like RF and BRF are tuned with grid search while the more advanced models are tuned with optuna to use it's bayesian search algorithm because just using randomized search is not reliable and using grid search is too exhaustive and the computational costs are too much
	 - **Ensemble Method:** A 	`StackingClassifier` was chosen as the final ensemble strategy due to its ability to learn the optimal combinations of the base model predictions
	 - **Meta-Learner:** `LogisticRegression` was selected and tuned as the meta learner for the Stacking ensemble

---

## Implementation

### Technologies Used

- **Python**: The core programming language for all development.
- **Jupyter Notebook**: For exploratory data analysis, data preprocessing, model building, and training (see `AoL_ML_V7_Final.ipynb`).
- **Streamlit**: For building an interactive web app (`app.py`) that allows users to input patient data and receive heart disease risk predictions.
- **pandas, numpy**: For data manipulation and numerical operations.
- **joblib, pickle**: For saving and loading trained machine learning models and preprocessing pipelines as `.pkl` files.

### App Structure (`app.py`)

The Streamlit app loads the exported preprocessing objects and trained model (all in `.pkl` format via joblib) and performs the following steps:
1. **User Input**: Collects patient data via UI.
2. **Preprocessing**: Applies scaling, imputation, feature engineering, and encoding to match model requirements.
3. **Feature Selection**: Retains only the features used by the final model.
4. **Model Prediction**: Uses the stacking classifier to predict probabilities for each heart disease stage.
5. **Output**: Displays the prediction and probabilities to the user.

### Key Implementation Details

- **Note on Input Features** : In the Streamlit app, certain columns—namely `ca`, `slope`, and `thal`—are commented out and not included as user input fields. This is because these columns had more than 30% missing values in the dataset, making them unreliable for modeling. As a result, they were removed during preprocessing and are not used in the final model.
- All data transformations (scaling, imputation, encoding) and the model itself are exported from the notebook using `joblib` for consistency and reproducibility.
- The app expects these files in the current directory:
  - `mmscalerBeforeFeatureEngineering.pkl`
  - `knnImputer.pkl`
  - `mmscalerAfterFeatureEngineering.pkl`
  - `ohEncoder.pkl`
  - `FinalModelStacking.pkl`

### Deployment

The application is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) using the `app.py` file directly from this GitHub repository. This enables anyone to access and interact with the app online without any local setup with the link https://ml-uci-heart-disease-jssaqbpnsr5e7ijuq5zqun.streamlit.app/.  

---

## Code & Usage

 1. Clone the repository:
     ```bash
     git clone https://github.com/Eylam65/ML-UCI-Heart-Disease.git
     cd <your-repository-name>
     ```
 2. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
 4. Open and run the main analysis notebook (`AoL_ML_V7_Final.ipynb`) in a Jupyter environment to see the data loading, preprocessing, model training, tuning, and evaluation steps
 5. If you just want to use the app locally, you can open and run the main app (`app.py`) by typing this on the bash(In Visual Studio Code, it uses git bash) :
    ```bash
     streamlit run app.py
     ```

---

## Conclusion
### Final Model Architecture
The best performing model was a `StackingClassifier` comprising four diverse base learners (Random Forest , Balanced Random Forest, XGBoost, and CatBoost - *but also consider LGBM here if your final tuned version makes the cut for your own ensemble*) and a tuned `LogisticRegression` as its meta-learner
### Models Explored
**Note:** the CV score here means that it's the F1 Score of the Train Data using Cross Validation (CV)
**Additional Note:** the CV score unless stated from optuna is manual CV score
- LGBM: explored, initial performance without specific samplers was noted. Tuned version (CV F1 ~0.488) was considered for ensembling. This model wasn't used mainly because of our lateness in trying this model out, and our decision is that the model's score needs to be at least 50% cv for it's to be worth to tune the Stacking again)
-   RF: used as base model (CV: ~0.462)
-   BRF: used as base model (CV: ~0.463)	
-   CatBoost: used as base model (best individual performance by far, "~50.4% F1 Macro" from optuna, and manual CV: ~46%)
-   XGB: used as base model (CV: ~0.461)
-   Logistic Regression: used as meta learner (tuned effectively within the Stacking ensemble framework)
-   RF (as meta-learner): explored briefly but did not outperform Logistic Regression.
### Individual Model Performance (on Test Set - Scoring Metric: F1_Macro Score)
- Random Forest: 0.51
- Balanced Random Forest: 0.50
- CatBoost: 0.50
- XGB: 0.45
### Final Model Performance
This is the classification report and confusion matrix in the final model:
```
			precision    recall  f1-score   
		0       0.78      0.83      0.81        83           
		1       0.56      0.53      0.54        53           
		2       0.47      0.41      0.44        22           
		3       0.41      0.43      0.42        21    
accuracy                            0.64       179   
macro avg       0.56      0.55      0.55       179
weighted avg    0.64      0.64      0.64       179
[[69 10  1  3] 
 [14 28  5  6] 
 [ 3  6  9  4] 
 [ 2  6  4  9]]
```
### Key Learnings from the Modeling Process
-   Every model, even though uses the same dataset prefers different sampler. Though, TomekLinks is very very much prefered (used in 3/4 in the base models)
-   Using ensembling is very effective, it's able to boost the very best individual model (59% accuracy and 51% macro) into 64% accuracy and 55% macro albeit the need to tune the ensemble a bit
-   Using Stacking is more preferred than Voting since besides catboost, other models performance is kinda shaky, even though Stacking needs to be set up first. Even though Stacking needs more setup, but because of the algorithm's ability to learn the optimal weight of each models makes it more effective than the simpler averaging approach of Voting
-   Picking the meta learner is also a challange since this project has a relative quick time limit, we needed a good base model that can be good with minimal tuning needed
-   Tuning the ensemble (stacking) in and of itself is a challange since it's needed to simmulate the entire Stacking environment for every iteration so it can take a long time
-   Learned to use new libraries (branching out to just using sklearns and imblearns which can be seen at the start), what's new is optuna, which is very handy in needing to tune very complex models such as catboost and xgboost, and obviously ensembling (Stacking)
-   Even though if we calculate only the yes/no proportion being 6:4, if we want to predict the stage of the heart disease in it's own, it's imbalanced (4:2:1:1)

---

## Limitations
-   **Time:** very limited, especially since the need to learn the basics while doing this project (can be seen by how many iterations this project is -> already V7). This limits the depth that can be explored for some more avenues
-   **Time management:** allocating time to this project is very hard since if the device is used for training the model, the device is unable to be used, which in hindsight, makes optuna such a godsend since it cuts so much time needed for this project because of it's optimization of grid search, making it less random than RandomizedSearchCV while also significantly faster than GridSearchCV using it's bayesian search method
-   **Device limitations:** the device used is not the top of the line spec of today, which is one of the reason this project took so long
-   **Low quantity of data:** because of the dataset having minority classes (1-3), the need to artificially sample the dataset is a problem since doing this may resulted in a loss of information (not a 'real' case), if the dataset is updated into a more balanced dataset, the performance of each class should increase also
-   **The need to drop 1 class:** this dataset actually has 5 classes, but because of just having 28 data, we decided to drop it because we can't get enough information to predict the class
-   **Order of ID Column Removal in Preprocessing:** The 'ID' column was removed during a later feature selection phase. For stricter adherence to best practices and to eliminate any potential for the ID to influence data transformations (such as the predictive imputation model for categorical features), it would have been ideal to remove this identifier at the very onset of the data preprocessing pipeline. But due to project timelines, re-running the entire modeling process with this revised order was not feasible for this iteration unfortunately

---

## Future Work & Recommendations
-   Definitely will try other datasets/projects and a much more challenging one at that
-   Or actually when our skills are better, trying this dataset again seems fun by implementing newer skillsets that we are not aware of right now, such things as:
	-   Further feature engineering: given the theory of how significant feature engineering can be, more can be done such as creating a more advanced feature selection (tailored to minimizes class 2 and 3 weaknesses) can get better results
	-   Exploring different meta learners: since time is at the essence right now, we can only try out base rf as the alternative and tune only with lr, exploration of a more complex meta learner such as a tuned rf, lgbm, or other models can yeild a better result too
	-   Data augmentation for minority classes: getting more data or a more refined handling for minority classes (mainly 2 and 3)
	-   a More Refined Preprocessing Workflow: Ensure identifier columns are unequivocally removed at the earliest stage of data preparation, prior to any imputation or feature engineering steps, to maintain the strictest data hygiene.

---

## Team Members and Contributions
### Ardelle Jody Nathaniel  
**Role: Lead Machine Learning Engineer & Technical Writer**  
Ardelle led the development of the final machine learning model and authored the project's detailed documentation. Out of all the models created by the team, Ardelle's model was selected due to its superior performance (Accuracy: **0.64**, F1 Macro Score: **0.55**).  

**Key Contributions:**
- Designed the complete machine learning pipeline: preprocessing, feature engineering, modeling
- Built and tuned multiple models: **Random Forest**, **Balanced Random Forest**, **XGBoost**, **CatBoost**, and **LightGBM**
- Applied balancing techniques: **SMOTETomek** and **TomekLinks**
- Built a **StackingClassifier** with **Logistic Regression** as the meta-learner
- Performed hyperparameter tuning using **Optuna**
- Wrote the full technical documentation in **README.md**

---

### Kenneth Andrew Lukita  
**Role: Modeling Support, LDA Clustering, & Presentation Lead**  
Kenneth contributed to model experimentation and supported the data exploration process. He also handled the design and delivery of the presentation materials.

**Key Contributions:**
- Conducted **baseline model training** and early evaluations using Jupyter Notebook
- Performed **clustering using Linear Discriminant Analysis (LDA)** to explore feature separation
- Helped validate models through metrics such as F1 Score, accuracy, and confusion matrices
- Created the **PowerPoint presentation**
- Summarized the technical and analytical results for effective communication

---

### Tandri Wibowo  
**Role: Streamlit Developer, PCA Clustering, & Deployment Engineer**  
Tandri was responsible for building and deploying the project’s user interface using **Streamlit**, and conducting additional exploratory analysis.

**Key Contributions:**
- Developed the interactive **Streamlit app** (`app.py`) to use the final model via `.pkl` file
- Successfully **deployed the application** on **Streamlit Cloud**
- Performed **clustering using Principal Component Analysis (PCA)** to visualize dimensionality-reduced patterns
- Created **visualizations** such as:
  - **Feature correlation heatmap**
  - **Feature importance plots** from Random Forest
- Conducted **hyperparameter tuning** for Random Forest during early modeling
