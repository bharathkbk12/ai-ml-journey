## Week 1 Reflection
This week was all about building the right foundation for ML work in Python. I focused on decorators, validation utilities, timing functions, and object-oriented design, and I started seeing how these small building blocks connect to real machine learning projects. Working through classes like Dataset, BaseModel, TrainingConfig, and preprocessing helpers helped me understand how data, model logic, and configuration fit together in a clean workflow.

### What I learned
- Decorators are a powerful way to add behavior such as timing and validation without cluttering the core logic.
- Object-oriented design helps keep ML code modular, reusable, and easier to extend.
- A simple project can still reflect real ML workflows when data, preprocessing, and training are separated cleanly.
- Writing tests early is essential for validating model behavior and catching configuration errors before they grow.

### What was hard
- Organizing the code across multiple modules without overengineering the project.
- Applying decorators correctly while preserving function behavior and metadata.
- Connecting small Python concepts to real ML ideas such as training, prediction, and reusable pipelines.

### Key insight
- A strong Python foundation is the backbone of every successful ML project. Clean abstractions make experimentation faster, safer, and easier to scale.

### Week 2 preview
- NumPy arrays and vectorized math
- Pandas for EDA and summarization
- Missing values, correlations, and data quality checks

## Week 2 Reflection
This week changed my perspective on machine learning because it showed me that a lot of the work happens before the model is even trained. I spent time with NumPy and Pandas, and it became clear that understanding data deeply is just as important as choosing an algorithm. The more I worked with vectorized operations, summaries, and exploratory checks, the more I saw how critical clean data and strong EDA are to making reliable predictions.

### What I learned
- NumPy array operations are the foundation of efficient ML computation and vectorized data processing.
- Matrix math and dot products are essential for understanding model behavior and neural network-style computations.
- Pandas tools like describe(), groupby(), isnull(), and corr() are core parts of exploratory data analysis.
- Missing values, outliers, and correlations often matter more than the model choice in the early stages of ML work.
- Data splitting and scaling are necessary steps before training or comparing models.

### What was hard
- Understanding broadcasting, shapes, and vectorized operations in real ML workflows.
- Moving from raw numeric calculations to meaningful EDA interpretation without jumping too quickly into modeling.
- Recognizing that noisy or incomplete data can hurt model performance more than the algorithm itself.

### Key insight
- EDA is not a side task; it is a major part of machine learning. If the data is noisy, missing, or misleading, the model will reflect that weakness.

### Week 3 preview
- Matplotlib advanced plots
- Feature engineering
- Scikit-learn basics and model evaluation


## Week 3 Reflection

### What I learned
- fit/predict/transform is the universal ML pattern in scikit-learn workflows
- Feature engineering often matters more than model choice, especially for tabular data like Titanic
- A Pipeline is essential because it prevents data leakage and keeps preprocessing and modeling consistent
- Accuracy alone is not enough; precision, recall, F1, and ROC-AUC provide a more honest view of model quality
- Cross-validation gives a more reliable estimate of performance than a single train/test split
- Cleaning missing data and encoding categorical variables correctly is a major part of the ML pipeline

### What was hard
- Understanding the difference between leakage and valid preprocessing during model training
- Making sure the feature engineering and encoding matched the model’s expected inputs
- Interpreting metrics beyond accuracy and realizing that class balance affects model evaluation
- Debugging warnings and shape mismatches when building a pipeline with a scaler and classifier

### Best result
- Titanic model: 0.788 test accuracy, 0.725 F1-score, 0.831 ROC-AUC
- Cross-validation mean accuracy: 0.795
- Training accuracy: 0.983, showing the model learned well without overfitting too severely

### Week 4 preview
- Andrew Ng ML Specialization (supervised learning deep dive)
- Project 1: Customer Churn Predictor (real Kaggle dataset)
- XGBoost + more advanced feature engineering