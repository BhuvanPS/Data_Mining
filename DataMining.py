# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %% [markdown]
# # **Analysis of Diabetes and Cardiovascular Disease Risk Factors Using NHANES Data**
# **Author:** *Bhuvan Purushothaman Subramani*  
# **Student Number:** *224113776*  
# **Email:** *s224113776@deakin.edu.au*  
# **Program:** *SIT731*
#
# ---
#
# ## Introduction
#
# This report presents an exploratory analysis of the **National Health and Nutrition Examination Survey (NHANES) 2017–2020** data, focusing on the relationship between **diabetes** and **cardiovascular disease (CVD) risk factors**.
#
# Our aim is to uncover patterns and associations between these chronic conditions and various **demographic**, **lifestyle**, and **clinical** variables. By analyzing these relationships, we seek to identify key risk factors that may contribute to the onset and progression of diabetes and CVD, thereby supporting early intervention strategies and public health planning.
#
# Key areas of focus include:
# - Age, gender, and ethnicity distribution
# - Physical activity and dietary habits
# - Blood pressure, cholesterol levels, and BMI
# - Prevalence of diagnosed diabetes and indicators of CVD
#
# This analysis will leverage **interactive visualizations** to make complex relationships more intuitive and to enhance insight discovery.
#

# %%
# Import Libraries
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, push_notebook
from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.layouts import row, column
import numpy as np

output_notebook()  # Enable Bokeh to display plots in the notebook

# Load Datasets from SAS files (assuming they are in the same directory)
demo_df = pd.read_sas('P_DEMO.XPT')        # Demographic data
diq_df = pd.read_sas('P_DIQ.XPT')          # Diabetes questionnaire data
bpx_df = pd.read_sas('P_BPXO.XPT')         # Blood pressure examination data
tch_df = pd.read_sas('P_TCHOL.XPT')       # Total cholesterol data
bmx_df = pd.read_sas('P_BMX.XPT')         # Body measures examination data
ins_df = pd.read_sas('P_INS.XPT')         # Insurance coverage data
ghb_df = pd.read_sas('P_GHB.XPT')         # Glycohemoglobin (HbA1c) data
paq_df = pd.read_sas('P_PAQ.XPT')         # Physical activity questionnaire data
alq_df = pd.read_sas('P_ALQ.XPT')         # Alcohol use questionnaire data
dr1_df = pd.read_sas('P_DR1TOT.XPT')      # Dietary recall data - first day total

# %%
# Data Merging (using for loop)
# Load DataFrames into a list
dataframes = [
    demo_df, diq_df, bpx_df, tch_df, bmx_df, ins_df, ghb_df, paq_df, alq_df, dr1_df
]

# Initialize merged_df with the first DataFrame
merged_df = dataframes[0].copy()

# Merge the remaining DataFrames in a loop
for df in dataframes[1:]:
    merged_df = pd.merge(merged_df, df, on='SEQN', how='inner')

print(merged_df.shape)

# %%

# Update selected columns with the correct moderate physical activity variable
selected_columns = [
    'SEQN',
    'RIDAGEYR',
    'RIAGENDR',
    'RIDRETH1',
    'DIQ010',             # Ever told you had diabetes?
    'BPXOSY1', 'BPXODI1',     # Systolic and diastolic BP
    'LBXTC',              # Total Cholesterol
    'BMXBMI',             # Body Mass Index
    'LBXIN',              # Insulin
    'LBXGH',              # Glycohemoglobin (HbA1c)
    'PAQ670',
    'ALQ111',             # Alcohol Use Frequency
    'DR1TCALC', 'DR1TSODI', 'DR1TTFAT', 'DR1TSUGR' # Dietary Intake
]

# Create a new DataFrame with the updated selected columns
df_subset = merged_df[selected_columns].copy()

# %%

# Check for missing values in the selected subset
missing_values = df_subset.isnull().sum()
print("Missing values per column:")
print(missing_values)
df_subset.shape


# %% [markdown]
# * PAQ670 has 2637 missing values, which is more than half of the dataset (over 59% missing). This could indicate that this variable is either not collected for most of the participants or perhaps is recorded only for specific groups.
# * BPXOSY1, BPXODI1, LBXTC, LBXIN, LBXGH, DR1TCALC, DR1TSODI, DR1TTFAT, and DR1TSUGR also have a significant proportion of missing values. These columns likely contain measurements or survey responses and could either be problematic due to non-response or incomplete data collection.

# %%
# Drop the PAQ670 column
df_subset = df_subset.drop(columns=['PAQ670'])

# %% [markdown]
# * The column `PAQ670` (moderate recreational activities) has been dropped from the `df_subset` DataFrame due to the substantial number of missing values. This will simplify further analysis by focusing on variables with more complete data.

# %%
missing_values = df_subset.isnull().sum()
print("\nMissing values per column (after dropping PAQ670):")
print(missing_values)

# %%

# Drop rows where key cardiovascular/diabetes-related variables are missing.
# We are targeting these key variables because missing data in these columns
# would lead to incomplete or unreliable analysis on diabetes and cardiovascular risk factors.
columns_to_check = ['BPXOSY1', 'BPXODI1', 'LBXTC', 'LBXGH', 'BMXBMI']
df_cleaned = df_subset.dropna(subset=columns_to_check)  # Drop rows where any of the specified columns are missing


# Display the shape of the cleaned DataFrame to see how many rows and columns remain after removing rows with missing data.
print(f"Shape of the DataFrame after handling missing values in key columns: {df_cleaned.shape}")

# Check for any remaining missing values in the cleaned DataFrame to ensure there are no missing values in other columns.
remaining_missing = df_cleaned.isnull().sum()  # Count the missing values in each column of the cleaned DataFrame
print("\nRemaining missing values per column:")
print(remaining_missing)  # Output the count of missing values in each column


# %%
# Drop rows where either LBXIN or ALQ111 is missing
df_cleaned = df_cleaned.dropna(subset=['LBXIN', 'ALQ111'], how='any').copy()

# %% [markdown]
#
# * Handling Missing Values in LBXIN and ALQ111
#
# * Rows where either Insulin (`LBXIN`) or Alcohol Use (`ALQ111`) values were missing have been removed from the DataFrame.

# %%
# Check remaining missing values
remaining_missing_final = df_cleaned.isnull().sum()
print("\nRemaining missing values per column (after dropping rows where either LBXIN or ALQ111 were missing):")
print(remaining_missing_final)

# %%

# Impute missing dietary values with the median (handling FutureWarning)
dietary_cols = ['DR1TCALC', 'DR1TSODI', 'DR1TTFAT', 'DR1TSUGR']

for col in dietary_cols:
    median_val = df_cleaned[col].median()
    df_cleaned[col] = df_cleaned[col].fillna(median_val)

# Imputing Missing Dietary Values

# Missing values in the dietary intake columns (Energy, Sodium, Total Fat, Total Sugars) have been imputed using the median value of each respective column. The median is a robust measure that is less affected by extreme values. The code has been updated to avoid a future warning from pandas.


# Check remaining missing values
remaining_missing_final_imputed = df_cleaned.isnull().sum()
print("\nRemaining missing values per column (after imputation):")
print(remaining_missing_final_imputed)

# Display the first few rows of the final cleaned and imputed DataFrame
print("\nFirst few rows of the final cleaned and imputed DataFrame:")
print(df_cleaned.head())

# %%

# Rename columns
new_column_names = {
    'SEQN': 'ParticipantID',
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Gender',
    'RIDRETH1': 'RaceEthnicity',
    'DIQ010': 'EverToldDiabetes',
    'BPXOSY1': 'SystolicBP',
    'BPXODI1': 'DiastolicBP',
    'LBXTC': 'TotalCholesterol',
    'BMXBMI': 'BMI',
    'LBXIN': 'Insulin',
    'LBXGH': 'HbA1c',

    'ALQ111': 'EverHad12DrinksYear',
    'DR1TCALC': 'EnergyKcal',
    'DR1TSODI': 'SodiumMg',
    'DR1TTFAT': 'TotalFatGm',
    'DR1TSUGR': 'TotalSugarsGm'
}

df_renamed = df_cleaned.rename(columns=new_column_names)


# Display the first few rows with the new column names
print(df_renamed.head())


# %% [markdown]
# * The columns in the DataFrame have been renamed to more descriptive and easier-to-understand names.

# %%

# Data Transformations for Categorical Variables 

# Gender Transformation: Map the numerical values (1.0 and 2.0) to descriptive labels ('Male' and 'Female')
gender_map = {1.0: 'Male', 2.0: 'Female'}
df_renamed['Gender'] = df_renamed['Gender'].map(gender_map)

# EverToldDiabetes Transformation: Map the numerical values (1.0 and 2.0) to descriptive labels ('Yes' and 'No')
diabetes_map = {1.0: 'Yes', 2.0: 'No'}
df_renamed['EverToldDiabetes'] = df_renamed['EverToldDiabetes'].map(diabetes_map)

# RaceEthnicity Transformation: Map the numerical values to descriptive categories for race/ethnicity
race_ethnicity_map = {
    1.0: 'Mexican American',
    2.0: 'Other Hispanic',
    3.0: 'Non-Hispanic White',
    4.0: 'Non-Hispanic Black',
    5.0: 'Other Race'
}
df_renamed['RaceEthnicity'] = df_renamed['RaceEthnicity'].map(race_ethnicity_map)

# EverHad12DrinksYear Transformation: Map the numerical values (1.0 and 2.0) to descriptive labels ('Yes' and 'No')
drink_map = {1.0: 'Yes', 2.0: 'No'}
df_renamed['EverHad12DrinksYear'] = df_renamed['EverHad12DrinksYear'].map(drink_map)

# Display the first few rows with the transformed columns
# This will allow us to visually inspect the changes and confirm that the categorical transformations were applied correctly.
print(df_renamed.head())


# %%

from bokeh.models import ColumnDataSource, Select, CustomJS
from bokeh.palettes import Category10


risk_factors = ['Age', 'BMI', 'DiastolicBP']
diabetes_status = 'Yes'
diabetes_column = 'EverToldDiabetes'
hist_data_yes = {}

# Create ColumnDataSource for each factor with proper bar positioning
for factor in risk_factors:
    data = df_renamed[df_renamed[diabetes_column] == diabetes_status][factor].dropna()
    hist, edges = np.histogram(data, bins=20)
    
    # Compute bin centers and widths
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_widths = edges[1:] - edges[:-1]  # Usually uniform

    hist_data_yes[factor] = ColumnDataSource(data=dict(
        x=bin_centers,
        top=hist,
        width=bin_widths
    ))

def get_max_y(factor):
    return max(hist_data_yes[factor].data['top']) * 1.1

initial_factor = risk_factors[0]

p = figure(height=700, width=600,
           title=f'Distribution of {initial_factor} among people with diabetes',
           y_axis_label='Frequency')

# ✅ Use center x and correct width
hist_renderer = p.vbar(x='x', top='top', bottom=0, width='width',
                       source=hist_data_yes[initial_factor],
                       fill_color=Category10[3][0], alpha=0.7,
                       legend_label='No. of people with Diabetes')

p.y_range.start = 0
p.y_range.end = get_max_y(initial_factor)
p.xaxis.axis_label = initial_factor
p.legend.location = "top_right"

select = Select(title="Select Risk Factor:", value=initial_factor, options=risk_factors)

callback = CustomJS(args=dict(
    sources_hist=hist_data_yes,
    hist_renderer=hist_renderer,
    p=p,
    xaxis=p.xaxis[0]
), code="""
    const factor = cb_obj.value;
    hist_renderer.data_source.data = sources_hist[factor].data;
    p.title.text = `Distribution of ${factor} among people with diabetes`;
    xaxis.axis_label = factor;
    const maxHist = Math.max(...sources_hist[factor].data['top']);
    p.y_range.end = maxHist * 1.2;
""")

select.js_on_change('value', callback)

show(column(select, p))


# %% [markdown]
# **Analysis of Distributions Among People with Diabetes**
#
# Based on the histograms:
#
# * **Age:** The distribution of age among people with diabetes in this dataset is skewed to the right, indicating a higher prevalence of older individuals. There's a notable concentration in the 60-70 and 75+ age ranges.
#
# * **BMI:** The Body Mass Index (BMI) distribution is also skewed to the right, suggesting that individuals with diabetes in this sample tend to have higher BMIs, with a peak in the obese range (approximately 30-35).
#
# * **DiastolicBP:** The distribution of diastolic blood pressure is roughly bell-shaped, centered around the 70-80 mmHg range, indicating a typical distribution with some variability.
#
# **Overall Conclusion:**
#
# In this specific dataset of individuals with diabetes, older age and higher BMI are common characteristics. Diastolic blood pressure is distributed around a central value. These observations highlight potential associations between these factors and diabetes within this population.

# %%

from bokeh.models import ColumnDataSource, CategoricalColorMapper, CustomJS

# Prepare the full data source
full_source = ColumnDataSource(df_renamed)

# This will be filtered
source = ColumnDataSource(data=dict(Age=[], TotalCholesterol=[], Gender=[]))

# Set columns
x_col = 'Age'
y_col = 'TotalCholesterol'
color_col = 'Gender'

# Unique gender values
genders = df_renamed[color_col].unique().tolist()

# Choose a palette
palette = Category10[max(3, len(genders))]

# Color mapper
color_mapper = CategoricalColorMapper(factors=genders, palette=palette)

# Create the plot
p = figure(
    x_axis_label=x_col,
    y_axis_label=y_col,
    title=f'Total Cholesterol vs. Age, Colored by Gender',
    tools="pan,wheel_zoom,box_zoom,reset,save,hover"
)

# Scatter plot
p.scatter(
    x=x_col,
    y=y_col,
    source=source,
    legend_field=color_col,
    color={'field': color_col, 'transform': color_mapper},
    size=8,
    alpha=0.6
)

p.legend.location = "top_right"

# Select widget
select = Select(title="Select Gender:", value="Both", options=["Both"] + genders)

# JavaScript callback for filtering
callback = CustomJS(args=dict(source=source, full_source=full_source, select=select), code="""
    const gender = select.value;
    const data = full_source.data;
    const new_data = { Age: [], TotalCholesterol: [], Gender: [] };

    for (let i = 0; i < data['Age'].length; i++) {
        if (gender === "Both" || data['Gender'][i] === gender) {
            new_data['Age'].push(data['Age'][i]);
            new_data['TotalCholesterol'].push(data['TotalCholesterol'][i]);
            new_data['Gender'].push(data['Gender'][i]);
        }
    }
    source.data = new_data;
""")

select.js_on_change('value', callback)

# Trigger initial display (equivalent to "Both")
source.data = dict(full_source.data)


# Layout
layout = column(select, p)
show(layout)


# %% [markdown]
# **Conclusion:**
#
# * Wide cholesterol spread across ages.
# * Weak visual age-cholesterol trend.
# * No obvious gender cholesterol difference.
# * Significant cholesterol variability at each age.
# * Outliers present for both genders.
# * No strong linear correlation evident.
# * Subtle gender differences possible.
# * Further statistical analysis needed.

# %%

from bokeh.models import CustomJSTickFormatter, Label
from bokeh.palettes import Vibrant3 as colors
import pandas as pd

# Bin the ages
bins = range(0, int(df_renamed['Age'].max()) + 10, 10)
labels = [f'{i}-{i+9}' for i in bins[:-1]]
df_renamed['age_group'] = pd.cut(df_renamed['Age'], bins=bins, labels=labels, right=False)

# Aggregate by age group and gender
grouped = df_renamed.groupby(['age_group', 'Gender'], observed=True).agg(
    mean_systolic=('SystolicBP', 'mean'),
    mean_diastolic=('DiastolicBP', 'mean')
).dropna().reset_index()

# Extract gendered BP data
def extract_gender_bp(data, column):
    female = data[data['Gender'] == 'Female'].set_index('age_group')[column].reindex(labels).fillna(0)
    male = data[data['Gender'] == 'Male'].set_index('age_group')[column].reindex(labels).fillna(0)
    return female, male

female_sys, male_sys = extract_gender_bp(grouped, 'mean_systolic')
female_dia, male_dia = extract_gender_bp(grouped, 'mean_diastolic')

# Reuse y_range
y_range = labels[::-1]  # Reverse age groups top to bottom

# Plot Systolic BP
p_sys = figure(
    title="Mean Systolic BP by Age Group and Gender",
    height=500, width=600,
    y_range=y_range,
    x_range=(-160, 160),
    x_axis_label="Systolic BP (mmHg)",
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

p_sys.hbar(y=y_range, right=female_sys.values, height=0.4, color=colors[0], legend_label="Women", line_width=0)
p_sys.hbar(y=y_range, right=-male_sys.values, height=0.4, color=colors[1], legend_label="Men", line_width=0)

p_sys.xaxis.formatter = CustomJSTickFormatter(code="return Math.abs(tick);")
p_sys.ygrid.grid_line_color = None
p_sys.yaxis.axis_label = "Age Group"
p_sys.legend.location = "bottom_left"
p_sys.legend.orientation = "vertical"

# Add gender labels at the top bar
p_sys.add_layout(Label(x=-120, y=0, text="Men", text_color=colors[1], text_font_size="10pt"))
p_sys.add_layout(Label(x=120, y=0, text="Women", text_color=colors[0], text_font_size="10pt"))

# Plot Diastolic BP
p_dia = figure(
    title="Mean Diastolic BP by Age Group and Gender",
    height=500, width=600,
    y_range=p_sys.y_range,
    x_range=(-120, 120),
    x_axis_label="Diastolic BP (mmHg)",
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

p_dia.hbar(y=y_range, right=female_dia.values, height=0.4, color=colors[0], legend_label="Women", line_width=0)
p_dia.hbar(y=y_range, right=-male_dia.values, height=0.4, color=colors[1], legend_label="Men", line_width=0)

p_dia.xaxis.formatter = CustomJSTickFormatter(code="return Math.abs(tick);")
p_dia.ygrid.grid_line_color = None
p_dia.yaxis.axis_label = "Age Group"
p_dia.legend.location = "bottom_left"
p_dia.legend.orientation = "vertical"

# Add gender labels at the top bar
p_dia.add_layout(Label(x=-90, y=0, text="Men", text_color=colors[1], text_font_size="10pt"))
p_dia.add_layout(Label(x=90, y=0, text="Women", text_color=colors[0], text_font_size="10pt"))

# Display the plots
show(row(p_sys, p_dia))


# %% [markdown]
# **Analysis of Mean Blood Pressure by Age Group and Gender**
#
# **Mean Diastolic BP**
#
# * For most age groups, the mean diastolic BP is slightly higher for women (orange bars) compared to men (blue bars).
# * This difference appears more pronounced in the older age groups (50-59, 60-69, 70-79).
# * Both genders show a general trend of increasing mean diastolic BP with age, although the increase for women seems a bit steeper in later years.
# * The 0-9 age group shows a relatively lower mean diastolic BP for both genders.
#
# **Mean Systolic BP**
#
# * Similar to diastolic BP, the mean systolic BP tends to be slightly higher for women in the older age groups (50-59, 60-69, 70-79).
# * Both men and women exhibit a clear trend of increasing mean systolic BP with age.
# * The increase in mean systolic BP with age appears more substantial than that of diastolic BP for both genders.
# * The youngest age group (0-9) has the lowest mean systolic BP for both sexes.
#
# **Overall Conclusion:**
#
# Mean systolic and diastolic blood pressure generally increase with age for both men and women. In the older age categories, women tend to have slightly higher mean systolic and diastolic blood pressure compared to men in this data. The most substantial increase with age is observed in systolic blood pressure for both genders.

# %%
from bokeh.models import CustomJS, TextInput, Button, TableColumn, DataTable

# Format DataFrame
table_columns = ['ParticipantID', 'Age', 'Gender', 'EverToldDiabetes', 'BMI', 'HbA1c']
df_table = df_renamed[table_columns].copy()
df_table['BMI'] = df_table['BMI'].round(1)
df_table['HbA1c'] = df_table['HbA1c'].round(1)

# Sources
source = ColumnDataSource(df_table)
original_source = ColumnDataSource(df_table)

# Table setup
columns = [TableColumn(field=col, title=col) for col in df_table.columns]
data_table = DataTable(source=source, columns=columns, width=800, height=400)

# Filter widgets
id_input = TextInput(title="Filter by Participant ID:")
gender_select = Select(title="Filter by Gender:", value="All", options=["All", "Male", "Female"])
diabetes_select = Select(title="Filter by Diabetes:", value="All", options=["All", "Yes", "No"])
reset_button = Button(label="Reset", button_type="warning")
export_button = Button(label="Download CSV", button_type="primary")

# CustomJS callback for filtering
filter_callback = CustomJS(args=dict(
    source=source,
    original=original_source,
    id_input=id_input,
    gender_select=gender_select,
    diabetes_select=diabetes_select
), code="""
    const id_val = id_input.value.trim().toLowerCase();
    const gender_val = gender_select.value;
    const diabetes_val = diabetes_select.value;
    
    const data = source.data;
    const orig = original.data;

    for (let key in data) data[key] = [];

    for (let i = 0; i < orig['ParticipantID'].length; i++) {
        const id = String(orig['ParticipantID'][i]).toLowerCase();
        const gender = orig['Gender'][i];
        const diabetes = orig['EverToldDiabetes'][i];

        const match_id = !id_val || id.includes(id_val);
        const match_gender = gender_val === 'All' || gender === gender_val;
        const match_diabetes = diabetes_val === 'All' || diabetes === diabetes_val;

        if (match_id && match_gender && match_diabetes) {
            for (let key in data) {
                data[key].push(orig[key][i]);
            }
        }
    }
    source.change.emit();
""")

# Attach callbacks
id_input.js_on_change('value', filter_callback)
gender_select.js_on_change('value', filter_callback)
diabetes_select.js_on_change('value', filter_callback)

# Reset callback
reset_callback = CustomJS(args=dict(
    source=source,
    original=original_source,
    id_input=id_input,
    gender_select=gender_select,
    diabetes_select=diabetes_select
), code="""
    source.data = Object.assign({}, original.data);
    id_input.value = '';
    gender_select.value = 'All';
    diabetes_select.value = 'All';
    source.change.emit();
""")
reset_button.js_on_click(reset_callback)

# CSV export callback
export_callback = CustomJS(args=dict(source=source), code="""
    const data = source.data;
    const cols = Object.keys(data);
    const nrows = data[cols[0]].length;
    
    let csv = cols.join(",") + "\\n";
    for (let i = 0; i < nrows; i++) {
        let row = cols.map(col => data[col][i]);
        csv += row.join(",") + "\\n";
    }

    // Trigger download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "filtered_data.csv";
    link.click();
""")
export_button.js_on_click(export_callback)

# Layout and show
controls = row(id_input, gender_select, diabetes_select, reset_button, export_button)
layout = column(controls, data_table)
show(layout)


# %%

from bokeh.layouts import gridplot

# Create AgeGroup column
bins = list(range(20, 81, 10))
labels = [f"{left}-{right}" for left, right in zip(bins[:-1], bins[1:])]
df_renamed['AgeGroup'] = pd.cut(df_renamed['Age'], bins=bins, right=False, labels=labels)

# Dietary columns to plot
dietary_cols = ['EnergyKcal', 'SodiumMg', 'TotalFatGm', 'TotalSugarsGm']
plots = []

# Loop through each dietary column
for i, col in enumerate(dietary_cols):
    # Group values into lists per age group
    grouped = df_renamed.groupby('AgeGroup', observed=True)[col].apply(lambda x: x.dropna().tolist())

    # Compute box plot statistics
    stats = grouped.apply(lambda x: {
        'q1': np.percentile(x, 25),
        'q2': np.percentile(x, 50),
        'q3': np.percentile(x, 75),
        'iqr': np.percentile(x, 75) - np.percentile(x, 25),
        'upper': min(max(x), np.percentile(x, 75) + 1.5 * (np.percentile(x, 75) - np.percentile(x, 25))),
        'lower': max(min(x), np.percentile(x, 25) - 1.5 * (np.percentile(x, 75) - np.percentile(x, 25)))
    })

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats.tolist(), index=stats.index).reset_index()
    stats_df['AgeGroup'] = stats_df['AgeGroup'].astype(str)  # Convert for x-axis FactorRange
    source = ColumnDataSource(stats_df)

    # Create the figure
    p = figure(x_range=stats_df['AgeGroup'], height=300, width=400,
               title=f"{col} by Age Group",
               x_axis_label="Age Group", y_axis_label=col,
               tools="pan,wheel_zoom,box_zoom,reset")

    # Draw box (q1 to q3)
    p.vbar(x='AgeGroup', width=0.7, bottom='q1', top='q3', source=source,
           fill_color=Category10[10][i % 10], line_color='black')

    # Draw median
    p.segment(x0='AgeGroup', y0='q2', x1='AgeGroup', y1='q2', source=source,
              line_color='black', line_width=2)

    # Draw whiskers
    p.segment(x0='AgeGroup', y0='upper', x1='AgeGroup', y1='q3', source=source,
              line_color='black')
    p.segment(x0='AgeGroup', y0='lower', x1='AgeGroup', y1='q1', source=source,
              line_color='black')

    plots.append(p)

# Arrange in a grid (2 columns)
grid = gridplot(children=plots, ncols=2)
show(grid)


# %% [markdown]
# **Conclusions:**
#
# * Energy, total fat medians stable across ages.
# * Sodium, total sugar medians slightly decrease with age.
# * Variability similar for energy, fat, sugars across ages.
# * Sodium variability seems lower in older groups.
# * Wide intake range within each age group.
# * Outliers present for all nutritional variables.
# * Subtle trends observed, not dramatic shifts.
# * Statistical tests needed for significance.

# %%

from bokeh.models import HoverTool
from bokeh.models import FactorRange

# Prepare ethnicity options
ethnicities = df_renamed['RaceEthnicity'].dropna().unique().tolist()
ethnicities.sort()
initial_ethnicity = ethnicities[0]

# Function to create grouped data (percentages)
def get_ethnicity_percentage_data(df, ethnicity):
    df_ethnicity = df[df['RaceEthnicity'] == ethnicity]
    diabetes_counts = df_ethnicity['EverToldDiabetes'].value_counts().reindex(['Yes', 'No'], fill_value=0)
    alcohol_counts = df_ethnicity['EverHad12DrinksYear'].value_counts().reindex(['Yes', 'No'], fill_value=0)

    diabetes_total = diabetes_counts.sum()
    alcohol_total = alcohol_counts.sum()

    categories = [('Diabetes', 'Yes'), ('Diabetes', 'No'), ('Alcohol', 'Yes'), ('Alcohol', 'No')]
    percentages = [
        round((diabetes_counts['Yes'] / diabetes_total * 100), 1) if diabetes_total else 0,
        round((diabetes_counts['No'] / diabetes_total * 100), 1) if diabetes_total else 0,
        round((alcohol_counts['Yes'] / alcohol_total * 100), 1) if alcohol_total else 0,
        round((alcohol_counts['No'] / alcohol_total * 100), 1) if alcohol_total else 0,
    ]
    colors = [Category10[4][0], Category10[4][0], Category10[4][1], Category10[4][1]]

    return pd.DataFrame({'category': categories, 'percentage': percentages, 'color': colors})

# Initial data
initial_data = get_ethnicity_percentage_data(df_renamed, initial_ethnicity)
source = ColumnDataSource(initial_data)

# Plot setup
p = figure(
    x_range=FactorRange(*initial_data['category']),
    height=400, width=700,
    title=f"Diabetes and Alcohol by Ethnicity: {initial_ethnicity}",
    x_axis_label="Condition", y_axis_label="Percentage (%)",
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

p.vbar(x='category', top='percentage', width=0.8, color='color', source=source)

# Hover tool
p.add_tools(HoverTool(tooltips=[("Category", "@category"), ("Percentage", "@percentage%")]))

# Dropdown
select_ethnicity = Select(title="Select Ethnicity:", value=initial_ethnicity, options=ethnicities)

# Callback for JS
callback = CustomJS(args=dict(
    source=source,
    df_json=df_renamed.to_json(orient='records'),
    plot=p,
    color1=Category10[4][0],
    color2=Category10[4][1]
), code="""
    const data = JSON.parse(df_json);
    const ethnicity = cb_obj.value;
    const filtered = data.filter(row => row['RaceEthnicity'] === ethnicity);

    let diabetes_yes = 0, diabetes_no = 0;
    let alcohol_yes = 0, alcohol_no = 0;

    for (let row of filtered) {
        if (row['EverToldDiabetes'] === 'Yes') diabetes_yes++;
        if (row['EverToldDiabetes'] === 'No') diabetes_no++;
        if (row['EverHad12DrinksYear'] === 'Yes') alcohol_yes++;
        if (row['EverHad12DrinksYear'] === 'No') alcohol_no++;
    }

    const diabetes_total = diabetes_yes + diabetes_no;
    const alcohol_total = alcohol_yes + alcohol_no;

    const diabetes_yes_pct = diabetes_total ? Math.round((diabetes_yes / diabetes_total) * 1000) / 10 : 0;
    const diabetes_no_pct = diabetes_total ? Math.round((diabetes_no / diabetes_total) * 1000) / 10 : 0;
    const alcohol_yes_pct = alcohol_total ? Math.round((alcohol_yes / alcohol_total) * 1000) / 10 : 0;
    const alcohol_no_pct = alcohol_total ? Math.round((alcohol_no / alcohol_total) * 1000) / 10 : 0;

    source.data = {
        category: [['Diabetes', 'Yes'], ['Diabetes', 'No'], ['Alcohol', 'Yes'], ['Alcohol', 'No']],
        percentage: [diabetes_yes_pct, diabetes_no_pct, alcohol_yes_pct, alcohol_no_pct],
        color: [color1, color1, color2, color2]
    };

    plot.title.text = "Diabetes and Alcohol by Ethnicity: " + ethnicity;
""")

# Attach callback
select_ethnicity.js_on_change('value', callback)

# Show layout
show(column(select_ethnicity, p))


# %% [markdown]
# **Analysis of Diabetes and Alcohol by Ethnicity**
#
# The images present bar charts showing the percentage of individuals within different ethnicities who have diabetes ("Yes" for Diabetes) versus those who do not ("No" for Diabetes), and similarly for alcohol consumption ("Yes" for Alcohol) versus no alcohol consumption ("No" for Alcohol).
#
# **Observations Across Ethnicities:**
#
# * **Diabetes:** Across all presented ethnicities (Mexican American, Non-Hispanic Black, Non-Hispanic White, Other Hispanic, Other Race), the percentage of individuals without diabetes ("No") is consistently and substantially higher than the percentage of those with diabetes ("Yes").
# * **Alcohol:** Similarly, across all ethnicities, the percentage of individuals who consume alcohol ("Yes") is notably higher than the percentage of those who do not ("No").
#
# **Specific Observations by Ethnicity:**
#
# * **Mexican American:** Approximately 20% have diabetes, and about 90% consume alcohol.
# * **Non-Hispanic Black:** Roughly 17% have diabetes, and about 90% consume alcohol.
# * **Non-Hispanic White:** About 14% have diabetes, and approximately 95% consume alcohol. This group appears to have the lowest percentage of diabetes and the highest percentage of alcohol consumption among those shown.
# * **Other Hispanic:** Around 17% have diabetes, and about 90% consume alcohol.
# * **Other Race:** Approximately 16% have diabetes, and about 83% consume alcohol. This group shows a slightly lower percentage of alcohol consumption compared to the others.
#
# **Conclusion:**
#
# Within each of the examined ethnicities, the prevalence of not having diabetes is much higher than having it, and the prevalence of alcohol consumption is considerably higher than not consuming alcohol. Non-Hispanic Whites in this data exhibit the lowest reported percentage of diabetes and the highest percentage of alcohol consumption compared to the other groups. The "Other Race" category shows a slightly lower percentage of alcohol consumption.

# %% [markdown]
# ## Summary of Exploratory Data Analysis
#
# Throughout this exploration, we visualized various aspects of the dataset, aiming to uncover potential relationships and distributions. We examined patterns in blood pressure related to age and gender, explored dietary intake through histograms and box plots, and investigated the prevalence of conditions like diabetes and alcohol consumption across different demographic groups. The interactive table provided a means to inspect individual records, while scatter plots helped us look for correlations between continuous variables.
#
# These initial visualizations offer a foundation for deeper analysis. Future steps could involve statistical testing to validate observed trends, incorporating a wider range of variables available in the dataset, and building interactive tools for more user-driven exploration. Additionally, predictive modeling could be employed to forecast health outcomes based on the patterns identified in this exploratory phase.
