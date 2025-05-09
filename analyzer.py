import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
from datetime import datetime

# --- Elérési utak ---
base_path = Path("D:/School/SZE/ml_jobs_analysis")
data_path = base_path / "data" / "jobs.csv"

# --- Időbélyeges output mappa ---
now_str = datetime.now().strftime("%Y%m%d-%H%M")
output_dir = base_path / "output" / f"analysis-{now_str}"
output_dir.mkdir(parents=True, exist_ok=True)
output_xlsx = output_dir / "analysis.xlsx"

# --- Adat előfeldolgozása CSV beolvasás előtt ---
df = pd.read_csv(data_path, index_col=0)

# --- Előkészítés ---
df.dropna(subset=['job_description_text', 'seniority_level', 'job_title'], inplace=True)

# --- Top skillek kigyűjtése ---
skill_counts = {}
for skills in df['seniority_level']:
    for skill in skills.split(','):
        skill = skill.strip()
        if skill:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
top_skills = pd.Series(skill_counts).sort_values(ascending=False).head(15)

# --- Top városok ---
top_locations = df['company_address_locality'].value_counts().head(10)

# --- Tapasztalati szintek ---
top_levels = df['seniority_level'].value_counts().head(10)

# --- Wordcloud a leírásokból ---
text_blob = " ".join(df['job_description_text'].astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_blob)

# --- Excel fájl írása diagramokkal ---
with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as writer:
    top_skills.to_frame(name='Count').to_excel(writer, sheet_name='Top Skills')
    top_locations.to_frame(name='Count').to_excel(writer, sheet_name='Top Locations')
    top_levels.to_frame(name='Count').to_excel(writer, sheet_name='Experience Levels')

    # Ábrák rajzolása és Excelbe illesztése
    workbook = writer.book

    # Top Skills chart
    fig, ax = plt.subplots()
    top_skills.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title('Top 15 Required Skills')
    fig.tight_layout()
    chart_path = output_dir / "top_skills.png"
    fig.savefig(chart_path)
    plt.close(fig)
    worksheet = writer.sheets['Top Skills']
    worksheet.insert_image('D2', str(chart_path))

    # Wordcloud mentése
    wc_path = output_dir / "wordcloud.png"
    wordcloud.to_file(wc_path)
    worksheet = writer.book.add_worksheet("Wordcloud")
    writer.sheets['Wordcloud'] = worksheet
    worksheet.insert_image('B2', str(wc_path))

    # --- MI modell: job_descr -> seniority osztályozás ---
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import numpy as np

    # Szűrd ki azokat a kategóriákat, amelyekből csak 1 db van
    counts = df['seniority_level'].value_counts()
    valid_classes = counts[counts > 1].index
    df_clf = df[df['seniority_level'].isin(valid_classes)][['job_description_text', 'seniority_level']].dropna()
    X = df_clf['job_description_text']
    y = df_clf['seniority_level']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Szöveg vektorizálása
    vectorizer = CountVectorizer(stop_words='english', max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Modell tanítása
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_vec, y_train)

    # Predikció és kiértékelés
    y_pred = clf.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Eredmények mentése Excelbe
    report_df.to_excel(writer, sheet_name='ML Classification Report')

    # Confusion matrix ábra mentése
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix: Description -> Level')
    fig_cm.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)
    worksheet_cm = writer.book.add_worksheet("Confusion Matrix")
    writer.sheets['Confusion Matrix'] = worksheet_cm
    worksheet_cm.insert_image('B2', str(cm_path))

print(f"Elemzes kesz. Kimenet: {output_xlsx}")

print(df['seniority_level'].value_counts()[df['seniority_level'].value_counts() == 1])
