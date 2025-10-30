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
1. **Select Data Source**: From the sidebar, select either of the options to load the file of your choosing. Refer above for synthetic dataset generator tool. IMDB urls are also supported.
2. **Post Data Load Panels**: Display Data, EDA, Cleaning & Pre-processing provides you with data manipulation capabilities. 
3. **Outputs**: Once satisfied, save a cleaned `parquet` version of the file. All saved paths are displayed on the UI.
4. **Custom Names**: You can provide custom names to your source and cleaned files, using the textbox provided, defaulted to `dataset` for source files and `cleaned` for processed files.

_**Note**: The app is designed to run locally_