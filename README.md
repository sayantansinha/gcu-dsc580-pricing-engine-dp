## User Guide
### ⚙️ Prerequisites & Setup
These are some pre-requisites that needs to be run or ensured before starting.

#### Python Installation
- Python **3.10 or higher** is recommended.
- Verify installation:

```bash
  python --version
```

#### Project Dependencies
Install required packages:

```bash
pip install -r requirements.txt
```

#### Tools - Synthetic Data Generation

The script `simulate_licensing_data.py` generates a synthetic dataset representing media licensing deals. The configuration file `tools\config\sim_config.toml` can be used to alter parameters that generate the data. The output file is generated at the location `data/source/licensing_deals.csv`. 
Use this file to upload to the application interface.

**Usage**

```bash
python ${PROJECT_ROOT}/tools/simulate_licensing_data.py
```

### ▶️ Running the App
From the project root directory, run Streamlit:

```bash
python -m streamlit run src/app.py
```

This will automatically start a local web server and open the app in your default browser.
If it doesn’t open automatically, navigate to:

http://localhost:8501

### 💡Navigating the App - User Guide
1. **Trigger New Pipeline**: From the sidebar, click the "New Pipeline" button to trigger a new pipeline process. Alternatively, you can also click on an existing pipeline run from the list above the button, which will load details regarding the same, and allow you to continue based on the stage the run currently is.
2. **Data Staging**: Select either of the options (file upload or URL) to load the file of your choosing. Refer above for synthetic dataset generator tool. Only IMDB urls are supported for feature master creation. Each data file loaded is staged, and preview is available to see data characteristics, along with schema validation output. 
4. **Feature Master**: Once files are staged, based on selection available, feature master needs to be built. Please note currently, only synthetic dataset serves as the base and IMDb data files are aggregated on top of that to build the final feature master. Therefore, the size of the synthetic dataset is what would be the eventual feature master size.
5. **Data Exploration Panels**: Display Data, EDA, Preprocessing (Cleaning) provides manipulation capabilities on the feature master file. Allows to inspect missing values, outliers, and variable distributions. Expanders reveal options for imputation, normalization, and encoding, with summary comparisons shown before and after cleaning. Once satisfied, save a cleaned `parquet` version of the feature master. Saved paths are displayed on the UI. It uses the `data` directory as the root under your project, make sure to check out the `local.env` file for the same.
5. **Analytical Tools (Modeling)**:  Enables users to select and execute predictive models (Linear Regression, Ridge, Gradient Boosting). Model metrics such as R-sqaured, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are automatically computed and displayed via interactive charts.
6. **Visual Tools**: Includes visualization and exploration utilities. Users can view correlation heatmaps, distribution charts, and engagement-level analyses across various features.
7. **Reporting**: After modeling, users can generate visual performance reports, download summaries in PDF or CSV format, and review metadata logs stored under `data/reports/<run_id>`.

_**Note**: The app is designed to run locally for now, with plans to deploy on a cloud platform like AWS in future.