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

### 💡Navigating the App
1. **Trigger New Pipeline**: From the sidebar, click the "New Pipeline" button to trigger a new pipeline process.
2. **Select Data Source**: Select either of the options to load the file of your choosing. Refer above for synthetic dataset generator tool. IMDB urls are also supported.
3. **Staged Data View**: Each data file loaded is staged, and preview is available to see data characteristics.
4. **Feature Master**: Once files are staged, based on selection available, feature master needs to be built. Please note currently, only synthetic dataset serves as the base and IMDb data files are aggregated to on top of that to build the final feature master. Therefore, the size of the synthetic dataset is what would be the eventual feature master size.
5. **Post Data Load Panels**: Display Data, EDA, Cleaning & Pre-processing provides manipulation capabilities on the feature master file.
5. **Cleaned Output**: Once satisfied, save a cleaned `parquet` version of the feature master. Saved paths are displayed on the UI. It uses the `data` directory as the root under your project, make sure to check out the `local.env` file for the same.

_**Note**: The app is designed to run locally only.