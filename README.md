# Reddit Depression Analytics  
Analisi distribuita di segnali di depressione in post Reddit tramite Big Data e NLP

## Descrizione del progetto
Questo progetto studia la presenza di **segnali linguistici riconducibili a stati depressivi** all’interno di post pubblicati su Reddit, utilizzando un approccio di **Big Data Analytics** basato su Apache Spark e tecniche di Natural Language Processing.

L’obiettivo non è costruire un classificatore clinico, ma **analizzare pattern testuali ricorrenti**, confrontare approcci di rappresentazione del testo e valutare la loro efficacia in un contesto distribuito e scalabile.

Il lavoro è stato svolto come progetto per il corso di *Modelli e Tecniche per Big Data*.

---

## Dataset
Il dataset utilizzato è una versione strutturata del **Reddit Self-reported Depression Diagnosis Dataset**, composto da post provenienti da:
- subreddit dedicati alla depressione
- subreddit di controllo (non-depressivi)

I dati sono forniti in formato XML e includono:
- identificativo utente
- contenuto testuale del post
- label binaria (depression / non-depression)

Il dataset presenta:
- linguaggio informale
- rumore semantico
- class imbalance

---

## Architettura della soluzione

### Pipeline di elaborazione
L’intera pipeline è stata progettata per funzionare in modo distribuito:

1. **Ingestione dati**
   - Parsing XML tramite Spark
   - Estrazione e normalizzazione dei campi rilevanti

2. **Preprocessing testuale**
   - Tokenizzazione
   - Rimozione stopword
   - Normalizzazione del testo
   - Gestione di post molto brevi o non informativi

3. **Feature Engineering**
   Sono stati confrontati due approcci principali:
   - TF-IDF
   - Word2Vec

4. **Training e valutazione**
   - Modelli supervisionati per classificazione binaria
   - Valutazione tramite precision, recall, F1-score

5. **Persistenza risultati**
   - Salvataggio risultati intermedi e finali
   - Creazione di viste aggregate per l’analisi

6. **Visualizzazione**
   - Frontend leggero in Streamlit
   - Consumo di viste precomputate per ridurre la latenza

---

## Modelli e approcci utilizzati

### TF-IDF + Logistic Regression
- Approccio baseline
- Buone prestazioni numeriche
- Sensibile al rumore lessicale
- Rischio di overfitting semantico

### Word2Vec + Random Forest
- Migliore rappresentazione semantica
- Maggiore robustezza al linguaggio informale
- Migliore capacità di generalizzazione
- Prestazioni complessive più stabili

La scelta finale privilegia **robustezza e interpretabilità** rispetto al solo incremento del punteggio F1.

---

## Analisi dei risultati
L’analisi dei risultati ha evidenziato:
- pattern linguistici ricorrenti associati alla depressione
- maggiore incidenza di termini legati a ruminazione e isolamento
- minore rilevanza di segnali legati all’apatia rispetto a quanto comunemente ipotizzato

I risultati sono coerenti con la letteratura scientifica di riferimento e sono stati interpretati in modo conservativo, evitando conclusioni cliniche.

---

## Scelte progettuali rilevanti

- Utilizzo di Spark per gestire:
  - grandi volumi di testo
  - preprocessing distribuito
  - feature extraction scalabile

- Separazione netta tra:
  - backend analitico (Spark)
  - frontend di visualizzazione (Streamlit)

- Precomputazione delle viste per:
  - ridurre la latenza
  - evitare carichi inutili sul cluster

Queste scelte riflettono una progettazione orientata a **scalabilità, manutenibilità e chiarezza architetturale**.

---

## Tecnologie utilizzate
- Apache Spark
- Python
- MLlib
- Word2Vec
- Scikit-learn
- Streamlit

---

## Limitazioni
- Il dataset non è rappresentativo della popolazione generale
- Le label sono auto-dichiarate dagli utenti
- Il progetto non ha finalità diagnostiche o cliniche
- L’analisi è limitata al testo dei post, senza contesto temporale esteso

---

## Possibili estensioni
- Analisi temporale dell’evoluzione linguistica degli utenti
- Integrazione con modelli transformer
- Valutazione delle performance su cluster di dimensioni diverse
- Introduzione di metriche di costo e tempo di esecuzione

---

## Autore
- Vito Simone Goffredo  

Progetto sviluppato nell’ambito del corso *Modelli e Tecniche per Big Data*, 2025.