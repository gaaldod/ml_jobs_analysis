# ML Álláshirdetés Elemző

## Rövid leírás
Ez a projekt egy Python-alapú eszköz, amely gépi tanulás és vizualizáció segítségével elemez USA gépi tanulás (ML) álláshirdetéseket.

## Használat

1. **Adatfájl elhelyezése**  
   Helyezd el a `jobs.csv` fájlt a projekt `data` mappájába.

2. **Futtatás**  
   Parancssorban futtasd az `analyzer.py` scriptet:
   ```
   python analyzer.py
   ```

3. **Eredmények**  
   Az eredmények az `output` mappában, egy időbélyeges almappában (`analysis-YYYYMMDD-HHMM`) jelennek meg.
   - Excel fájl: `analysis.xlsx` (több lappal: statisztikák, wordcloud, ML riportok)
   - PNG képek: diagramok, szófelhő, confusion matrix

## Korlátozások

- **Oszlopnevek fixek:**  
  A program csak a dokumentációban megadott oszlopnevekkel működik (`job_description_text`, `seniority_level`, stb.).

- **Bemeneti fájl helye:**  
  A `jobs.csv` fájlt a `data` mappában keresi.

- **Nincs grafikus felület:**  
  A program parancssorból futtatható, nincs grafikus felhasználói felület.

- **Adathalmaz elvárások:**  
  Csak akkor működik, ha a főbb mezők (állásleírás, tapasztalati szint, pozíció) nem hiányoznak.

- **Angol nyelvű adatok:**  
  Az elemzés és az ML modell angol nyelvű álláshirdetésekre van optimalizálva.

## Függőségek

- Python 3.8+
- pandas
- matplotlib
- seaborn
- wordcloud
- scikit-learn
- xlsxwriter

Telepítés:
```
pip install pandas matplotlib seaborn wordcloud scikit-learn xlsxwriter
```

## Szerző
Készítette: [Gaál Dominik - E0G9J5]

---

# ML Job Ads Analyzer

## Short Description
This project is a Python-based tool for analyzing US machine learning (ML) job ads using machine learning and visualization.

## Usage

1. **Place the data file**  
   Place the `jobs.csv` file into the `data` folder of the project.

2. **Run**  
   Run the `analyzer.py` script from the command line:
   ```
   python analyzer.py
   ```

3. **Results**  
   Results will appear in the `output` folder, in a timestamped subfolder (`analysis-YYYYMMDD-HHMM`).
   - Excel file: `analysis.xlsx` (with multiple sheets: statistics, wordcloud, ML reports)
   - PNG images: charts, wordcloud, confusion matrix

## Limitations

- **Column names are fixed:**  
  The program only works with the column names specified in the documentation (`job_description_text`, `seniority_level`, etc.).

- **Input file location:**  
  The script expects `jobs.csv` in the `data` folder.

- **No GUI:**  
  The program is command-line only, no graphical user interface.

- **Dataset requirements:**  
  The script requires that key fields (job description, seniority level, job title) are present and not missing.

- **English data only:**  
  The analysis and ML model are optimized for English job ads.

## Dependencies

- Python 3.8+
- pandas
- matplotlib
- seaborn
- wordcloud
- scikit-learn
- xlsxwriter

Install dependencies:
```
pip install pandas matplotlib seaborn wordcloud scikit-learn xlsxwriter
```

## Author
Created by: [Dominik Gaal - E0G9J5] 