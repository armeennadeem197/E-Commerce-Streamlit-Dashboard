# Interactive data analytics dashboard

This project contains a schema-aware Streamlit dashboard for CSV exploration, cleaning, EDA, visualizations, and data-driven business insights. The bundled ecommerce_sales_data.csv is loaded by default, while any compatible CSV can be uploaded from the sidebar. EDA.ipynb and the original dataset are preserved.

## Run with the installed system Python

This project intentionally does not create or use a .venv folder.

From Command Prompt in this folder, use the working system interpreter:

    C:\Users\home\AppData\Local\Python\bin\python.exe -m pip install -r requirements.txt
    C:\Users\home\AppData\Local\Python\bin\python.exe -m streamlit run app.py

If the py launcher is available on another Windows installation, these equivalent commands also work:

    py -m pip install -r requirements.txt
    py -m streamlit run app.py

If python is available on PATH, use:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Do not reinstall packages unnecessarily. Check the existing installation first with:

    C:\Users\home\AppData\Local\Python\bin\python.exe -c "import streamlit, pandas, numpy, matplotlib, seaborn, plotly; print('All dashboard libraries are available')"

## Dashboard sections

- Overview: source status, responsive KPI cards, staged Apply Filters and Reset Filters controls, rows remaining, active filters, data quality summary, and a correlation preview.
- Data understanding: head, tail, first/last rows, random sample, shape, dtypes, info output, missing values, column type detection, descriptive statistics, and an interactive table.
- Data cleaning: duplicate removal, missing-value drop/fill methods, numeric conversion, date/time parsing including %H:%M:%S, IQR outlier inspection/removal/capping, before-and-after quality cards, an audit trail, and cleaned or cleaned-and-filtered CSV downloads.
- EDA: feature engineering, univariate, bivariate, multivariate, and time-series analysis.
- Visualizations: a mandatory numerical correlation heatmap plus up to three additional Plotly charts at one time.
- Insights: selectable equivalent columns for advertising spend versus revenue, category versus metric, and category versus region. Insights state limitations and never claim causation from correlation.

The dashboard keeps the original uploaded DataFrame untouched in session state and applies cleaning actions to a separate working copy.
