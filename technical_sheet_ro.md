# COALITION-4 RECONVECT — Fișă tehnică

**RECONVECT** — **RE**current **CONVECT**ive nowcasting. Prognoză de foarte scurtă durată
pentru fenomene convective pe teritoriul României, adaptată după sistemul COALITION-4 al
MeteoSwiss: un encoder-forecaster multi-rezoluție antrenat pe date MTG FCI, radar OPERA și
descărcări electrice LINET, care estimează precipitațiile în 5 clase și apariția binară a
fulgerelor pe o grilă comună de 1 km.

```
repository   coalition4-rcnn   branch wp
grid         1536 x 768  EPSG:31700  (1 km)
cadență      pas master de 15 min
arhivă       2025-01-01 .. 2026-08-13
```

> Artefactele marcate **CRITIC** sunt consumate în etapele ulterioare. Documentul este
> întreținut împreună cu `README.md` și `technical_sheet_en.md`; toate trei se actualizează
> pe măsură ce sunt integrate funcționalități noi.

**Cuprins:** [1 Prezentare generală](#1-prezentare-generală-a-sistemului) · [2 Niveluri](#2-nivelurile-de-rezoluție) ·
[3 Moduri](#3-moduri-intrări-și-etichete) · [4 Produse de date](#4-produsele-de-date) ·
[5 Arhitectură](#5-arhitectura-și-parametrii-impliciți-de-antrenare) · [6 Praguri](#6-praguri-esențiale) ·
[7 Etape comune](#7-planul-de-execuție--etapele-comune) · [8 Ramura A](#8-ramura-a-antrenarea-completă-reconvect) ·
[9 Ramura B](#9-ramura-b-studiul-de-ablație-și-modelul-de-referință-sepconv-ens) ·
[10 Ramura C](#10-ramura-c-ansamblul-sezonier) · [11 Ramura D](#11-ramura-d-distilarea-cunoștințelor) ·
[12 Criticalitate](#12-criticalitatea-artefactelor)

---

## 1. Prezentare generală a sistemului

Trei instrumente sunt reproiectate pe aceeași suprafață de 1 km, decupate în patch-uri de
256 × 256 și furnizate unui encoder-forecaster a cărui arhitectură este **citită din setul
de date**, nu definită static în cod. Numărul de pași temporali anteriori și ulteriori,
nivelurile de intrare și numărul de canale provin integral din `metadata.json`; prin
urmare, modificarea ferestrei de secvență este o decizie asupra datelor, nu o modificare de
cod.

```
achiziție -> reproiecție -> perioada comună -> selecția patch-urilor -> construirea secvențelor
          -> decuparea patch-urilor -> statistici -> seturi TFRecord -> antrenare
          -> inferență -> validare -> raport
```

| Sarcină | Țintă |
|---|---|
| Precipitații, 5 clase | OPERA `rainfall_rate` discretizat la 10/20/30/40 mm/h |
| Precipitații, continuu | OPERA `rainfall_rate` |
| Apariția fulgerelor | LINET, apariție binară |

**Dezechilibrul claselor.** Marea majoritate a pixelilor nu prezintă precipitații
semnificative, iar clasele de intensitate ridicată sunt cu ordine de mărime mai rare. Din
acest motiv, fiecare funcție de pierdere din sistem este ponderată: un obiectiv neponderat
pe această distribuție se minimizează prin estimarea valorii de repaus în fiecare punct,
ceea ce obține un scor bun fără a prognoza nimic. Selecția patch-urilor este convectivă —
DBSCAN aplicat pe câmpul de precipitații, cu un prag configurabil (`--threshold`) — însă
*în interiorul* unui patch selectat contribuie fiecare pixel, inclusiv cei cu intensitate
redusă. Nu se aplică niciun prag la nivel de pixel.

| Element de stocare | Observație |
|---|---|
| MTG `.npy` per ciclu de repetare | 5 canale |
| MTG brut `.nc` per chunk | 2 chunk-uri per ciclu |
| Arhivă 7-Zip a unui set de date | `-mx=5` ajunge la 4,8 % din sursă, `-mx=1` la 11,5 % |
| Comprimare zstd a depozitelor `.npy` | nivel 10, aplicată direct pe fișiere, ~8,5x măsurat pe întreaga arhivă |

---

## 2. Nivelurile de rezoluție

Se recomandă consultarea rezoluției fizice, nu a denumirii nivelului.

| Nivel | Nativ | Patch | Pooling | Canale |
|---|---|---|---|---|
| `past_hr` (HR) | 1 km | 256 × 256 | fără | MTG `vis_06`; LINET `density`, `current`, `occurrence` |
| `past_mr` (LR) | 2 km | 128 × 128 | medie 2 × 2 | OPERA `reflectivity`, `rainfall_rate`; MTG `ir_38`, `ir_105`, `wv_63`, `wv_73` |

**HR desemnează nivelul de rezoluție ridicată, iar LR nivelul de rezoluție redusă.**
Patch-urile extrase sunt denumite `{variable}_{HHMM}_{HR|LR}.npy` în consecință. Tensorul
de intrare de rezoluție redusă poartă în continuare denumirea `past_mr` în cod și în
`metadata.json`, întrucât aceasta este înscrisă în fiecare checkpoint antrenat; se citește
ca nivelul LR.

**Reconcilierea nivelurilor de rezoluție.** (1) Se reproiectează mai întâi toate datele pe
aceeași grilă de 1 km, astfel încât un număr de patch să corespundă aceleiași porțiuni
geografice pentru toate produsele, iar alinierea între scări să fie efectuată o singură
dată per pas de cadență. (2) Pooling-ul se aplică exclusiv *descendent*, niciodată
ascendent — supraeșantionarea ar genera pixeli artificiali, iar modelul ar învăța artefacte
de interpolare.

---

## 3. Moduri: intrări și etichete

Denumirea modului indică direct sarcina: `_rainfall` = OPERA în 5 clase, `_continuous` =
regresie OPERA (varianta utilizată pentru comparația cu modelul de referință),
`_occurrence` = fulgere binar. Eticheta este întotdeauna HR (256 px), indiferent de
nivelurile de intrare.

| Mod | Intrări HR | Intrări LR | Etichetă |
|---|---|---|---|
| `mtg_opera_radar_only_rainfall` | `vis_06` | `opera_reflectivity` `opera_rainfall_rate` | precipitații, 5 clase |
| `mtg_opera_mtgmr_rainfall` | `vis_06` | + `ir_38` `ir_105` `wv_63` `wv_73` | precipitații, 5 clase |
| `mtg_lightning_opera_rainfall` | `density` `current` `occurrence` `vis_06` | + MTG IR/WV | precipitații, 5 clase |
| `mtg_opera_mtgmr_continuous` | `vis_06` | + MTG IR/WV | precipitații, regresie |
| `mtg_lightning_opera_occurrence` | `density` `current` `occurrence` `vis_06` | + MTG IR/WV | fulgere, binar |
| `mtg_opera_occurrence` † | `vis_06` | + MTG IR/WV | fulgere, binar |
| `opera_radar_only_rainfall` ‡ | `opera_rainfall_rate_hr` | — | precipitații, 5 clase |
| `opera_sepconv_logz` ‡ | `opera_rainfall_rate_hr` | — | precipitații `log_zscore` |

† Exclusiv model-student pentru distilare — nu poate fi construit cu `create_datasets.py`; se antrenează pe setul de date al modelului-profesor.
‡ Perechea de comparație cu modelul de referință, exclusiv radar prin concepție. Ambele preiau câmpul în HR la 256 px, astfel încât tensorii de intrare sunt identici; rezoluția de ieșire a modelului este dată de cea mai fină intrare a sa, iar acesta este singurul mod fără alt canal HR care să o mențină la 256.

Numărul de canale decurge direct: un tensor HR are forma `(T, 256, 256, n_hr)`, iar unul LR
`(T, 128, 128, n_lr)`. Un mod **fără intrări HR nu deține deloc tensorul `past_hr`** —
modelul se construiește din grupurile pe care setul de date le furnizează.

`opera_rainfall_rate` și `opera_rainfall_rate_hr` reprezintă același câmp la două niveluri:
intrarea de 2 km redusă la 128 px, respectiv același câmp reproiectat păstrat la
1 km / 256 px, întrucât acesta constituie eticheta. `opera_sepconv_logz` este singurul mod
care utilizează forma HR atât ca intrare, cât și ca etichetă.

**Clasele de precipitații (mm/h):** `0: R<10 · 1: 10–20 · 2: 20–30 · 3: 30–40 · 4: R≥40`

---

## 4. Produsele de date

| Produs | Canale | Cadență | Rezoluție | Script sursă |
|---|---|---|---|---|
| MTG FCI L1C | `vis_06` `ir_38` `ir_105` `wv_63` `wv_73` | 10 min | 1 km / 2 km | `pipeline_msg_mtg.py` |
| OPERA | `reflectivity` `rainfall_rate` | 15 min | 2 km | `pipeline_opera.py` |
| LINET | `density` `current` `occurrence` | 15 min | 1 km | `linet_export.py` |

MTG este livrat sub forma a 40 de body chunks per ciclu de repetare; teritoriul României se
regăsește în chunk-urile 35 și 36 (`ROMANIA_CHUNKS`), aplicate identic pe ruta NMA și pe cea
de Data Store. Descărcările electrice sunt agregate direct pe grila României, astfel încât
nu necesită etapa de reproiecție.

| `--source` | Comportament |
|---|---|
| `nma` | SFTP de pe serverul intern, pe fereastra solicitată |
| `datastore` | Preia exclusiv ciclurile enumerate în `mtg_missing_timesteps.json` |
| `both` | Mai întâi NMA, apoi Data Store pentru restul |
| `local` | Fără descărcare — extrage datele brute existente în `_raw_chunks/` |

Originea este înregistrată per ciclu în `provenance.json`, deoarece cele două surse livrează
fișiere cu denumiri native identice; ulterior, aceasta nu mai poate fi reconstituită din
date.

**Stocare pe mai multe discuri.** Depozitul MTG depășește capacitatea unui singur disc, la
~47 MB per ciclu. `--spill_dir` alternează între depozite ori de câte ori cel activ scade
sub `--min_free_gb`, criteriu reevaluat înaintea fiecărei ferestre **și în ambele sensuri**,
întrucât `--delete_raw` restituie spațiu chiar discului de pe care se citește.
`store_registry.py` consemnează pe ce disc a fost stocată fiecare dată; `reproject.py` și
`summarize_mtg.py` citesc toate rădăcinile înregistrate.

---

## 5. Arhitectura și parametrii impliciți de antrenare

| Parametru | Valoare | Domeniu de aplicare |
|---|---|---|
| Encoder | ResBlock + ConvGRU (`ResGRU`), canale `[32, 64, 128]` | toate |
| Decoder | `[128, 64, 32]` în ordine inversă, upsampling biliniar + conexiuni skip | toate |
| Ramuri de intrare | câte una per nivel, unificate la scările corespunzătoare | toate |
| Pași anteriori / ulteriori | citiți din `sequence_meta` (implicit 3/3) | toate |
| Optimizator | `Adam(lr=1e-3)` | etapa base |
| Pierdere (fulgere) | `WeightedFocalLoss(gamma=2.0)` | distribuție a priori din `lightning_fraction` |
| Pierdere (clasificarea precipitațiilor) | `WeightedFocalCategoricalCrossentropy` | distribuție a priori din `opera_rainfall_fraction` |
| Pierdere (regresia precipitațiilor) | MSE ponderat | comună cu SepConv |
| Epoci / batch | `20` / `32` | toate |
| Dropout / normalizare | `0.1` / `layer` | toate |
| Precizie mixtă | `true` (fp16) | toate |
| Program LR | `cosine_warmup`, 1e-3 → 1e-6, 3 epoci de încălzire | etapa base |
| Oprire timpurie | `val_loss`, patience 6, restaurarea celui mai bun model | toate |
| Finetune Swin | transformer Swin peste rețeaua îngheșată | etapa finetune |

Forecasterul funcționează **într-o singură trecere**: pornește de la zerouri repetate de
`future_timesteps` ori și rulează stiva ConvGRU pe întregul orizont, emițând toate cadrele
simultan. Nicio componentă din RECONVECT nu este autoregresivă.

---

## 6. Praguri esențiale

Tabelul complet se regăsește în `README.md`; mai jos sunt enumerate exclusiv valorile care
influențează rezultatele, nu prezentarea acestora.

| Constantă | Implicit | CLI | Rol |
|---|---|---|---|
| `DBSCAN_THRESHOLD` | 10 mm/h | `--threshold` | Pragul ratei de precipitații pentru selecția patch-urilor de antrenare |
| `DBSCAN_EPS` / `MIN_SAMPLES` | 5 px / 20 px | — | Raza clusterului și dimensiunea minimă |
| `RAINFALL_CLASS_EDGES` | 10/20/30/40 | — | Limitele celor 5 clase; modificarea impune reantrenarea |
| `DEFAULT_RAIN_LOW` / `HIGH` | 0.35 / 0.55 | `--rainfall_*` | Histerezis pe `p(argmax)` pentru clasele cu precipitații |
| `LIGHTNING_LOW_THRESHOLD` | 0.90 | `--lightning_low_threshold` | Pragul inferior al histerezisului pe câmpul de probabilități |
| `DEFAULT_STRIDE` | 128 px | `--stride` | Pasul de deplasare la inferența cu ponderare Hann; suprapunere 50 % |
| `DEFAULT_KD_ALPHA` / `TEMPERATURE` | 0.7 / 4.0 | `--kd_alpha` | Ponderea țintelor atenuate ale modelului-profesor și temperatura de atenuare |
| `alpha_max` | 100.0 | `[radar_loss]` | Plafonează ponderile per clasă; fără plafonare, fp16 devine instabil |
| `label_smoothing` | 0.01 | `[radar_loss]` | Previne `log(0)` la saturarea stratului de activare final al rețelei Swin |

---

## 7. Planul de execuție — etapele comune

Toate ramurile parcurg aceste etape. Fiecare intrare precizează ce realizează scriptul
**executat individual**, ce artefacte produce și ce script le preia ulterior. Un rezultat
marcat **CRITIC** este consumat în aval: **fluxul de procesare nu poate continua fără el.**
O etapă ulterioară se oprește cu eroare atunci când fișierul lipsește și — cazul cu
adevărat costisitor — se execută până la capăt pe un fișier care nu mai corespunde datelor
de pe disc, raportând rezultate pentru o arhivă diferită de cea utilizată efectiv.

### 0. Cadența master
```bash
python validate_timestep.py --step_minutes 15
```
- **Descriere** — stabilește pasul master și derivă filtrul de minute al fiecărui produs din `product_cadences.config`.
- **Scrie** — `our_data/timestep_config.json` **CRITIC**
- **Citit de** — toate scripturile de achiziție, `reproject`, `identify_patches`, `extract_patch_seq`, `extract_patches`, scripturile de sumarizare, determinarea perioadei comune. Modificarea pasului aici se propagă în toate etapele ulterioare.

### 1a. Achiziția MTG
```bash
python our_data/satellite_data/pipeline_msg_mtg.py --start ... --end ... --source nma|datastore|both|local
```
- **Descriere** — descarcă chunk-urile FCI 35/36 și extrage câte un fișier `.npy` per canal per ciclu de repetare. Execuția se desfășoară în valuri de `--workers` cicluri; cu `--delete_raw`, datele brute sunt eliberate val cu val, astfel încât volumul maxim de date brute aflate simultan pe disc corespunde unui val, nu întregului interval.
- **Scrie** — `MTG/<channel>/….npy`, `mtg_constants.json` **CRITIC**
- **Scrie** — `MTG/provenance.json` **CRITIC** — sursa fiecărui ciclu. Cele două surse utilizează denumiri de fișiere native identice, astfel încât originea nu mai poate fi reconstituită ulterior.
- **Separat** — `--source local` extrage datele brute deja existente pe disc, fără descărcare. `--delete_only` eliberează datele brute fără re-extragere. `--record_existing` atribuie proveniență datelor anterioare registrului.
- **Citit de** — `reproject.py`, `summarize_mtg.py`

### 1b. Achiziția OPERA
```bash
python our_data/opera_data/pipeline_opera.py --start ... --end ... --ssh_key ...
```
- **Descriere** — preia compozitele OPERA în format HDF5, reproducând ierarhia de date de pe serverul la distanță. Datele sunt păstrate în format nativ, astfel încât o corecție de proiecție să nu impună niciodată o nouă descărcare.
- **Scrie** — `our_data/opera_data/{reflectivity,rainfall_rate}/YYYY/MM/DD/*.h5` **CRITIC**
- **Citit de** — `reproject.py`, `summarize_opera_data.py`

### 1c. Exportul LINET
```bash
python our_data/lightning_data/linet_export.py --start ... --end ... --format kml
```
- **Descriere** — descarcă exporturile brute de descărcări electrice, câte un fișier KML pe zi.
- **Notă** — `--end` este **exclusiv** în acest script, spre deosebire de toate celelalte.
- **Scrie** — `<out>/kml_data/<date>/<date>.kml` **CRITIC**
- **Citit de** — `read_kml_version2.py`

### 2a. Agregarea descărcărilor electrice pe grilă
```bash
python our_data/lightning_data/read_kml_version2.py
```
- **Descriere** — agregă descărcările electrice direct pe grila României, la cadența etichetelor; întrucât agregarea le plasează deja acolo, reproiecția nu este necesară.
- **Scrie** — `lightning_data/{density,current,occurrence}/….npy` **CRITIC**
- **Scrie** — `filtered_out_reports/lightning_filtered_out_<date>.json` — descărcările eliminate pentru poziționare în afara grilei. Exclusiv pentru audit; nu este citit de niciun script.
- **Citit de** — `extract_patches`, `compute_normalization_stats`, `summarize_lightning_data`

### 2b. Reproiecția pe grila României
```bash
python reproject.py --all --workers 6
```
- **Descriere** — reeșantionare KD-tree a datelor MTG și OPERA pe suprafața comună de 1536 × 768. Ulterior, un număr de patch corespunde aceleiași porțiuni pentru toate produsele.
- **Scrie** — `our_data/reprojected_data/….npy` **CRITIC**
- **Scrie** — `romania_grid_lats.npy` / `_lons.npy` — definiția grilei, reutilizată la reprezentările grafice și la exportul NetCDF.
- **Scrie** — `reproject_<category>.log` — eșecurile, scăzute din manifest prin `--errors_log`, astfel încât o reproiecție eșuată să nu fie contabilizată ca prezentă.
- **Separat** — `--mtg_dir` citește un depozit aflat pe alt disc; în absența sa, se parcurge succesiv fiecare rădăcină din indexul depozitelor.
- **Citit de** — `identify_patches`, `extract_patches`, `compute_normalization_stats`

### 3. Sumarele de acoperire (câte unul per produs)
```bash
python our_data/<product>/summarize_<product>.py --start 2025-01-01 --end 2026-08-13 --chart
```
- **Descriere** — măsoară acoperirea pe zile pornind de la rezultatele **extrase**, nu de la descărcările brute: o reproiecție eșuată după o descărcare reușită este invizibilă la o scanare a datelor brute.
- **Scrie** — `our_data/<product>_data/<product>_summary.csv` și `<product>_missing_timesteps.json` **CRITIC** — alături de produsul pe care îl descriu, nu în rădăcina depozitului de cod, și raportate la locația scriptului, nu la directorul de lucru.
- **Grafic** — `<product>_coverage.png` cu `--chart`, în același director. Bare lunare cu o linie care unește vârfurile. Exclusiv de prezentare; nu este citit de niciun script.
- **Notă** — fișierul JSON al pașilor de timp lipsă pentru MTG constituie cererea de completare adresată Data Store: sunt preluate exact ciclurile enumerate acolo. Este necesară specificarea `--start`/`--end`; în caz contrar, o dată fără niciun fișier nu este raportată niciodată ca lipsă și nu poate fi niciodată solicitată.
- **Separat** — `summarize_mtg --npy_dir` acceptă mai multe rădăcini și le scanează ca pe o arhivă unică, pentru un depozit distribuit pe mai multe discuri. `--scan {npy,raw,reprojected}` alege întrebarea la care se răspunde: `npy` (implicit) descrie depozitul extras și stă la baza completării din Data Store, `raw` descrie datele sosite înainte de extragere, iar `reprojected` descrie exact ceea ce va citi `extract_patches`. Lista de lipsuri rezultată dintr-o scanare `reprojected` descrie lacune de reproiecție și **nu** trebuie transmisă către `--source datastore`.
- **Citit de** — `intersect_product_coverage`; `pipeline_msg_mtg --source datastore`

### 4. Determinarea perioadei comune de acoperire a produselor
```bash
python intersect_product_coverage.py --summary opera_rainfall_rate=our_data/opera_data/opera_summary.csv \
    [--summary mtg=our_data/satellite_data/mtg_summary.csv --summary lightning=our_data/lightning_data/lightning_summary.csv]
```
- **Descriere** — intersectează produsele solicitate, păstrând pașii de timp în care *toate* sunt disponibile. Produsele impuse se aleg pentru fiecare execuție în parte, astfel încât un model exclusiv radar să nu fie limitat de lacunele altui instrument.
- **Scrie** — `our_data/timestep_manifest.csv` **CRITIC** — pașii de timp pe care fiecare etapă ulterioară îi poate utiliza.
- **Grafic** — `our_data/intersect_summary.png` — linii lunare, câte una per categorie, raportate la fiecare motiv de omisiune.
- **Notă** — cele două câmpuri OPERA constituie chei separate; astfel, un model exclusiv de precipitații păstrează eșantioane pentru care reflectivitatea lipsește, iar un model care citește reflectivitate nu primește niciodată un pas de timp fără aceasta.
- **Citit de** — `extract_patch_seq_for_datasets`

### 5. Selecția patch-urilor convective
```bash
python identify_patches.py --start 2025-01-01 --end 2026-08-13
```
- **Descriere** — DBSCAN aplicat pe OPERA `rainfall_rate` (prag configurabil, implicit 10 mm/h; eps 5, min_samples 20), marcând care dintre cele 18 patch-uri este activ la fiecare pas de timp. Selectează **patch-uri, nu pixeli**.
- **Scrie** — `our_data/patch_index/patch_index.csv` și `.json` **CRITIC**
- **Grafic** — `plots/patches_<date>_<HHMM>.png` cu `--date --plot`, împreună cu un fișier NetCDF echivalent de ~28 MB fiecare. Exclusiv pentru diagnoză; `--purge_plots` le elimină.
- **Notă** — un singur index deservește toate perioadele, iar ordinea rândurilor sale definește axa de patch a matricelor salvate. O execuție cu `--date` nu suprascrie niciodată indexul principal.
- **Citit de** — `extract_patch_seq_for_datasets`, `extract_patches`, `data_statistics`

### 6. Ferestrele de secvență și împărțirea în seturi
```bash
python extract_patch_seq_for_datasets.py [--period LABEL --past N --future M --start ... --end ...]
```
- **Descriere** — construiește secvențe continue în timp și împărțirea pe blocuri Czibula (blocuri de 6 ore, 80/10/10).
- **Scrie** — `{train,validation,test}_data_<source>[_<period>].csv` **CRITIC**
- **Scrie** — `sequence_meta_<source>[_<period>].json` **CRITIC** — `past_steps`, `future_steps`, `step_minutes`. Prin acest fișier, orizontul modelului devine o proprietate a datelor.
- **Scrie** — `extract_patch_seq_drops_….csv` — fiecare candidat respins, împreună cu motivul respingerii.
- **Citit de** — `extract_patches`, `create_datasets`, `compute_normalization_stats`, distribuțiile a priori ale claselor, `verification_keys`, `data_statistics`

### 7. Decuparea patch-urilor
```bash
python extract_patches.py [--period LABEL] [--products opera ...]
```
- **Descriere** — decupează plăci de 256 × 256 din suprafețele complete; produsele LR sunt reduse prin average pooling la 128 px. Întotdeauna descendent, niciodată ascendent.
- **Scrie** — `our_data/patches/<date>/<var>_<HHMM>_{HR|LR}.npy` **CRITIC**
- **Notă** — rezultatul **nu** este sufixat cu perioada: toate perioadele scriu în același arbore, astfel încât o a doua perioadă reprezintă în mare parte o operațiune fără efect pe zona de suprapunere. Colecția este invalidată de reconstruirea fișierului `patch_index.csv`, niciodată de o perioadă nouă.
- **Scrie de asemenea** — `our_data/patches/<date>/_patch_index.json`, listele de patch-uri active din care a fost construită fiecare dată. Orice pas de timp a cărui listă s-a modificat între timp este re-extras, nu ignorat, deoarece un patch care devine activ se inserează în mijlocul listei și deplasează toate pozițiile ulterioare. **CRITIC**
- **Separat** — `--audit_pool` raportează ce date s-au desincronizat și nu extrage nimic.
- **Citit de** — `create_datasets`

### 8. Statisticile de normalizare
```bash
python compute_normalization_stats.py [--period LABEL] [--variables ...]
```
- **Descriere** — media și deviația standard per variabilă, calculate **exclusiv pe cheile de antrenare**.
- **Scrie** — `normalization_stats_<source>[_<period>].json` **CRITIC**
- **Notă** — invariantul este utilizarea acelorași constante la antrenare și la inversare. Antrenarea cu un set și inversarea cu altul conduc la valori mm/h eronate, cu o abatere care crește odată cu intensitatea — fără ca vreo eroare să fie semnalată.
- **Citit de** — `create_datasets`, `train_models`, `predict_full_domain`, `validate_predictions`, `sepconv_predict`, `evaluate_coalition`, `generate_report`

### 9. Distribuțiile a priori ale claselor
```bash
python opera_rainfall_fraction.py [--period LABEL]
python lightning_fraction.py [--period LABEL]
```
- **Descriere** — măsoară distribuția claselor în **propriul** set de antrenare al modelului.
- **Scrie** — `opera_rainfall_fraction_….json` / `lightning_fraction_….json` **CRITIC**
- **Notă** — domeniul și denumirea fișierului provin din același tag. O distribuție calculată pe o altă fereastră descrie un echilibru pe care modelul nu îl întâlnește niciodată, iar funcția de pierdere ponderată corectează în acest caz un dezechilibru diferit de cel real.
- **Citit de** — `train_models`, `sepconv_ensemble_training`

---

## 8. Ramura A: antrenarea completă RECONVECT

Modelul principal: toate cele trei modalități, precipitații în 5 clase. Continuă direct
după etapa 9.

### A1. Construirea setului de date
```bash
python create_datasets.py --mode mtg_lightning_opera_rainfall [--period LABEL] [--no-archive]
```
- **Scrie** — `our_data/datasets/<run_tag>/{train,validation,test}/*.tfrecord` **CRITIC**
- **Scrie** — `metadata.json` per partiție **CRITIC** — `input_shapes` și `label_shape`. Antrenarea își citește arhitectura din aceste fișiere, astfel încât o fereastră diferită nu impune nicio modificare de cod.
- **Notă** — fără `--no-archive`, setul de date este comprimat, iar shard-urile sunt eliminate după verificare; antrenarea ar necesita în acest caz o restaurare prealabilă.

### A2. Antrenarea modelului de bază
```bash
python train_models.py --config training.config --mode mtg_lightning_opera_rainfall --stage base
```
- **Descriere** — construiește encoder-forecasterul pe baza `metadata.json`. Restaurează automat un set de date arhivat.
- **Scrie** — `models/coalition_<run_tag>.keras` **CRITIC**
- **Scrie** — `models/history_<run_tag>.json` — mod, sursă, etapă, tip de etichetă, epoci, durată de execuție.
- **Scrie** — `models/coalition_<run_tag>.meta.json` **CRITIC** — perioada pe care a fost antrenat modelul, verificată înaintea analizei de importanță a caracteristicilor, astfel încât un model să nu fie niciodată explicat cu date pe care a fost antrenat.

### A3. Finetune Swin
```bash
python train_models.py --config training.config --mode mtg_lightning_opera_rainfall --stage finetune
```
- **Scrie** — `models/coalition_<run_tag>_finetuned.keras`

### A4. Inferență pe întregul domeniu
```bash
python predict_full_domain.py --mode ... --date YYYY-MM-DD
```
- **Descriere** — asamblează patch-uri suprapuse, ponderate Hann, într-o suprafață completă la `--stride 128` (suprapunere 50 %), eliminând discontinuitățile plăcilor de 256 px.
- **Scrie** — `inference/predict_<run_tag>/*.npy`, `*_hyst.npy` — salvate ca matrice, astfel încât o explorare a pragurilor să nu impună repetarea inferenței.
- **Grafic** — `*_hits.png`, `*_perclass_hits.png`

### A5. Validarea și calibrarea pragurilor
```bash
python validate_predictions.py --track rainfall --year Y --month M
```
- **Descriere** — parcurge luna în căutarea eșantioanelor cu cel puțin un pixel ≥ 10 mm/h, execută inferența și calibrează pragul superior al histerezisului per orizont de prognoză, prin maximizarea CSI-ului agregat.
- **Scrie** — `validation/rainfall_<Y>_<M>_summary.json` **CRITIC** — pragurile calibrate și blocul `per_patch`.
- **Scrie** — `…_samples.csv`
- **Grafic** — `…_metrics.png`; suprapuneri pe zile cu `--date`
- **Citit de** — `generate_report`, `build_patch_ensemble`, `bundle_eval_scores`

### A6. Figuri și raport
```bash
python visualize_gt_vs_pred.py --mode ...
python generate_report.py --year Y --month M
```
- **Scrie** — `full_domain_plots/…`, `validation/report_<Y>_<M>.pdf`

---

## 9. Ramura B: studiul de ablație și modelul de referință SepConv-ens

Două întrebări, separate în mod deliberat. Ablația evaluează contribuția **arhitecturii** la
intrări identice; analiza contribuției modalităților evaluează aportul MTG și al fulgerelor.
Ambele componente utilizează exclusiv date radar și **niciuna nu este autoregresivă**:
comparația se oprește la t+4, pe care compunerea îl construiește exclusiv din observații.

| Tag | Fereastră | Pași | Destinație |
|---|---|---|---|
| `w44` | past=4 / future=4 | 9 | `opera_sepconv_logz` — t+2 și t+4 citesc `t−4` |
| `w34` | past=3 / future=4 | 8 | `opera_radar_only_rainfall` — 4 cadre de intrare |

### B1. Ambele ferestre (repetarea etapei 6)
```bash
python extract_patch_seq_for_datasets.py --period w44 --past 4 --future 4 --start ... --end ...
python extract_patch_seq_for_datasets.py --period w34 --past 3 --future 4 --start ... --end ...
```
Ulterior, se repetă etapele 7–9 pentru fiecare perioadă.

### B2. Verificarea contaminării seturilor de date între două execuții
```bash
python verification_keys.py --write --reconvect_tag w34 --sepconv_tag w44
```
- **Descriere** — stabilește eșantioanele pe care două execuții împărțite independent pot fi comparate în mod corect. Intersectează cele două partiții de test, apoi elimină orice eșantion care apare în datele de antrenare sau de validare ale oricăruia dintre modele. Ferestre de secvență diferite plasează același eșantion pe părți opuse ale împărțirii, astfel încât, în lipsa acestui pas, fiecare model ar fi evaluat parțial pe date din care celălalt a învățat — și, în plus, pe o populație diferită.
- **Scrie** — `verification_keys_<source>_<a>_vs_<b>.json` **CRITIC** — denumirea consemnează perechea descrisă.
- **Notă** — se execută **înainte** ca oricare model să citească datele de test. `--sepconv_tag` este obligatoriu: în prezența mai multor ferestre pe disc, nu există o valoare implicită sigură.

### B3. Seturile de date
```bash
python create_datasets.py --mode opera_sepconv_logz        --period w44 --no-archive
python create_datasets.py --mode opera_radar_only_rainfall --period w34 --no-archive
```
- **Verificare** — ambele execuții trebuie să afișeze un fișier de statistici sufixat cu perioada și **fără** mențiunea `<- overridden`. Fiecare model este normalizat pe propria partiție, astfel încât `--global_stats` lipsește din ambele.

### B4. Antrenarea
```bash
python sepconv_ensemble_training.py --period w44
python train_models.py --config training.config --mode opera_radar_only_rainfall --period w34 --stage base
```
- **Descriere** — antrenează Bm1/Bm3/Bm5 (t+1, t+3, t+5 = 15/45/75 min) pentru modelul de referință, precum și ablația pe arhitectura RECONVECT.
- **Scrie** — `models/sepconv_<run_tag>_bm{1,3,5}.keras`, `history_sepconv_<run_tag>.json` **CRITIC**
- **Scrie de asemenea** — `models/checkpoints/<name>_latest.keras` după fiecare epocă, împreună cu un fișier `.json` care conține indexul epocii următoare. Ambele modele reiau execuția din acest punct; `--fresh` îl ignoră. Astfel, fiecare execuție lasă în urmă **două** stări: cele mai bune ponderi în fișierul final și ultima epocă în checkpoint.
- **Notă** — ambele citesc același `training.config`: `[defaults]` furnizează `epochs` și `batch_size` fiecăruia, iar secțiunea opțională `[sepconv]` le poate suprascrie, preluând `learning_rate` din `[lr_schedule].initial_lr` și `es_patience` din `[early_stopping].patience`. Prin urmare, o diferență între cele două modele nu poate proveni din bugetul de antrenare. *Programul* ratei de învățare nu este unificat în mod deliberat — RECONVECT utilizează cosine warmup, iar modelul de referință reproduce `ReduceLROnPlateau` din lucrarea originală.
- **Separat** — `--datasets_root` / `--output_dir` plasează setul de date și checkpoint-urile pe alt disc.

### B5. Evaluarea
```bash
python evaluate_sepconv_ensemble.py --period w44
python evaluate_coalition.py --mode opera_radar_only_rainfall --period w34
```
- **Descriere** — compune t+1…t+4 prin `sepconv_compose`, denormalizează cu statisticile propriei ferestre și discretizează în mm/h la aceleași limite de clasă utilizate de RECONVECT — astfel, cele două modele nu pot fi diferențiate prin praguri.
- **Scrie** — `evaluation/eval_sepconv_<run_tag>/evaluation_results.json`
- **Grafic** — `metrics_per_leadtime.png`; `--plot_samples N` reprezintă clasa observată față de cea estimată și valorile mm/h estimate.
- **Separat** — `--weights best|latest` alege care dintre cele două stări salvate este evaluată. `best` este fișierul final, iar `latest` este checkpoint-ul per epocă. Compararea lor arată dacă epocile ulterioare celei mai bune au condus la supraînvățare sau dacă oprirea timpurie a întrerupt o execuție care încă se îmbunătățea.

**Ambii evaluatori trebuie să primească același set de chei fixat.** **CRITIC**
Cele două ferestre sunt împărțite independent, astfel încât partiția de test proprie
fiecărui model reprezintă o populație diferită de a celuilalt *și* se suprapune peste
datele sale de antrenare (test w34: 5420, test w44: 5125, în comun 4745; 380 dintre cheile
de test ale modelului de referință au fost utilizate de RECONVECT la antrenare). Evaluarea
restrânsă la intersecție este ceea ce face cifrele comparabile:

```bash
python verification_keys.py --write --reconvect_tag w34 --sepconv_tag w44
python evaluate_sepconv_ensemble.py --period w44 \
    --verification_keys our_data/verification_keys_dbscan_w34_vs_w44.json
python evaluate_coalition.py --mode opera_radar_only_rainfall --period w34 \
    --verification_keys our_data/verification_keys_dbscan_w34_vs_w44.json
```

Filtrarea se realizează pe `(date, reference_utc, patch)`, valori consemnate în fiecare
shard, deci este exactă, nu pozițională. Seturile de date construite înainte de existența
acestor câmpuri nu corespund niciunei chei și sunt respinse cu indicația de reconstruire.

### B6. Analiza contribuției modalităților
```bash
python bundle_eval_scores.py
python feature_importance_analysis.py --model ... --data ... --methods gradcam_xi shap
```
- **Descriere** — consolidează mai multe execuții în fișiere CSV per orizont de prognoză, apoi aplică Grad-CAM/Xi, SHAP și Shapley clasic. Perechea de ablație compară două matrice Xi pentru a evidenția modul în care intrările rămase preiau rolul grupului eliminat.
- **Scrie** — `eval_leadtime-<prefix>-<letters>.csv`, `results/feature_importance/…`
- **Notă** — literele codifică intrările utilizate de o execuție: `o` = exclusiv OPERA, `om` = OPERA + MTG IR/WV.

---

## 10. Ramura C: ansamblul sezonier

Câte un membru per sezon, selectat per patch pe baza performanței măsurate. Setul de membri
se înregistrează o singură dată, iar fiecare etapă ulterioară se verifică în raport cu
această înregistrare, nu cu conținutul curent al discului.

### C1. Înregistrarea planului
```bash
python create_datasets.py --mode <mode> --ensemble [--seasons_config ...]
```
- **Descriere** — enumeră membrii pe baza definițiilor sezoanelor, raportează acoperirea și suprapunerile față de datele disponibile și adaugă planul rezultat.
- **Scrie** — `our_data/ensemble_registry.json` **CRITIC** — exclusiv prin adăugare; etapele ulterioare citesc ultima stare.
- **Notă** — un membru cu acoperire sub 90 % este raportat `PARTIAL`, dar rămâne construibil.

### C2. Construirea și antrenarea fiecărui membru
```bash
python create_datasets.py --mode <mode> --period 2025warm
python train_models.py --config training.config --mode <mode> --period 2025warm --stage base
```
- **Notă** — `--period` se rezolvă mai întâi din metadatele de secvență, apoi din registru; astfel, tag-urile de fereastră precum `w44` funcționează fără a fi membri înregistrați.
- **Verificare** — `python train_models.py --check-ensemble --mode <mode>` raportează membrii pentru care seturile de date sunt construite.

### C3. Evaluarea fiecărui membru per patch
```bash
python validate_predictions.py --track rainfall --year Y --month M --mode <mode>
```
- **Descriere** — produce blocul `per_patch` citit de mecanismul de selecție. Se execută o dată per membru.
- **Scrie** — `validation/…_summary.json` cu `per_patch` **CRITIC**

### C4. Selecția per patch
```bash
python build_patch_ensemble.py --mode <mode>
```
- **Descriere** — alege membrul cu cel mai bun scor pentru fiecare dintre cele 18 patch-uri. Exclusiv selecție — evaluarea a fost efectuată la C3.
- **Scrie** — `our_data/ensemble_manifest_<mode>_<source>.json` **CRITIC** — tabela de rutare, cu variante de rezervă pe sezon și pe modelul global pentru fiecare alocare.
- **Citit de** — `ensemble_inference`

### C5. Rutarea la inferență
`ensemble_inference.PatchEnsemble` asociază un patch membrului alocat, cu revenire la
membrul de sezon și apoi la modelul global.

---

## 11. Ramura D: distilarea cunoștințelor

Antrenează un model-student care funcționează **fără intrare de fulgere**, astfel încât
componenta de fulgere să poată fi utilizată și la datele la care LINET nu este disponibil.

### D1. Antrenarea modelului-profesor
```bash
python create_datasets.py --mode mtg_lightning_opera_occurrence
python train_models.py --config training.config --mode mtg_lightning_opera_occurrence --stage base
```
- **Descriere** — modelul-profesor primește stiva completă de intrări, inclusiv LINET `density`, `current`, `occurrence`.
- **Scrie** — `models/coalition_mtg_lightning_opera_occurrence_<source>.keras` **CRITIC**

### D2. Distilarea către modelul-student
```bash
python train_lightning_kd.py --teacher_mode mtg_lightning_opera_occurrence --student_mode mtg_opera_occurrence
```
- **Descriere** — antrenează pe setul de date al **modelului-profesor**, cu `past_hr` restrâns la ultimele `STUDENT_HR_CHANNELS` (= `vis_06`), astfel încât modelul-student nu vede niciodată date de fulgere. Funcția de pierdere combină țintele atenuate ale modelului-profesor la `--kd_alpha 0.7`, temperatura 4.0, cu valorile de referință.
- **Notă** — `mtg_opera_occurrence` **nu** poate fi construit cu `create_datasets`: există exclusiv ca model-student pe datele modelului-profesor.
- **Scrie** — `models/coalition_<student_run_tag>_kd.keras`, `history_…_kd.json` **CRITIC**

### D3. Validarea simultană a modelului-profesor și a modelului-student
```bash
python validate_predictions.py --track kd --year Y --month M
```
- **Descriere** — execută ambele modele pe eșantioane identice și calibrează histerezisul fiecăruia în mod independent, astfel încât comparația să nu fie un artefact al unui prag comun.
- **Scrie** — `validation/lightning_<Y>_<M>_kd_summary.json`
- **Grafic** — `…_kd_metrics.png`

---

## 12. Criticalitatea artefactelor

**Comune tuturor perioadelor.** Cinci artefacte nu poartă niciun tag de sursă sau de
perioadă și sunt utilizate de fiecare execuție; reconstruirea oricăruia dintre ele le
afectează pe toate simultan:

```
timestep_config.json      timestep_manifest.csv     patch_index.csv
our_data/patches/         reprojected_data/
```

**Etichetate per execuție.** Toate celelalte artefacte sunt denumite `<source>[_<period>]`,
unde `source` este întotdeauna `dbscan` (metoda DBSCAN de selecție a eșantioanelor — o
constantă, nu un parametru), iar `period` este eticheta opțională `--period`. `run_tag` are
forma `<mode>_<source>[_<period>]` și este construit de `build_run_tag()` — sursa unică de
adevăr pentru denumirile modelelor, ale checkpoint-urilor și ale seturilor de date.

Se precizează că `--source` din `pipeline_msg_mtg.py` nu are legătură cu acest mecanism:
acolo desemnează originea descărcării.

**Artefacte terminale — pot fi eliminate în siguranță.** Niciun script din aval nu le
citește; scriptul care le-a generat le regenerează:

| Artefact | Producător |
|---|---|
| `<product>_coverage.png` | cele trei scripturi de sumarizare |
| `intersect_summary.png` | `intersect_product_coverage` |
| `patch_index/plots/` și `plots/nc/` | `identify_patches --plot` |
| `mtg_store_distribution.png` | `store_registry --chart` |
| figurile din `inference/`, `full_domain_plots/` | `predict_full_domain`, `visualize_gt_vs_pred` |
| figurile din `evaluation/` | `evaluate_*` |
| `results/feature_importance/` | `feature_importance_analysis` |
| `validation/report_<Y>_<M>.pdf` | `generate_report` (livrabil) |
| `our_data/data_statistics/` | `data_statistics` |

### Desincronizarea colecției de patch-uri

`our_data/patches/` este utilizat în comun de toate perioadele — un fișier de patch depinde
exclusiv de (dată, oră, variabilă, rezoluție) și de `patch_index.csv`. O perioadă nouă nu îl
invalidează; reconstruirea fișierului `patch_index.csv` îl invalidează. **CRITIC**

Un fișier de patch este o matrice de plăci fără nicio consemnare a plăcii aflate pe fiecare
poziție; poziția *k* înseamnă „al *k*-lea patch activ la acest pas de timp". Când un patch
devine activ, acesta se inserează în mijlocul listei și deplasează toate pozițiile
ulterioare:

```
fișierul conține : [2, 3, 4,    7, 8, 9, 13, 14]
indexul indică   : [2, 3, 4, 5, 7, 8, 9, 13, 14]
                    ok ok ok  ^-- deplasate de aici
```

Doar ultima poziție iese din domeniu. Cele deplasate se citesc fără eroare și returnează
**placa greșită** — patch-uri aflate la ~1.000 km distanță pe domeniu — asociind intrarea
unei regiuni cu eticheta alteia. Întrucât `extract_patches` ignoră fișierele existente,
această stare persistă.

| Mecanism de protecție | Efect |
|---|---|
| `our_data/patches/<date>/_patch_index.json` | Consemnează lista de patch-uri active din care a fost construit fiecare fișier; pașii de timp desincronizați sunt re-extrași, nu ignorați. |
| `create_datasets.StalePatchPool` | O valoare `idx_t*` în afara domeniului generează eroare, în loc să fie completată cu zerouri. |

```bash
python extract_patches.py --audit_pool [--period TAG]
```

O variabilă *absentă* este în continuare completată cu zerouri — acest comportament este
corect. O poziție în afara domeniului nu este niciodată corectă. Suprascrierea unui fișier
desincronizat elimină și perechea sa `.npy.zst`.

**Fișierele desincronizate trebuie eliminate înainte de comprimarea colecției** —
comprimarea rescrie datele de modificare, iar pentru o colecție anterioară mecanismului de
consemnare acestea constituie singura dovadă a desincronizării.

### Localizarea datelor

Toate valorile implicite se raportează la depozitul de cod, nu la directorul de lucru,
astfel încât scripturile pot fi executate din orice locație, fără `cd`.

| Rădăcină | Parametru | Variabilă de mediu | Implicit |
|---|---|---|---|
| patch-uri, CSV-uri, statistici | `--data_root` | `COALITION4_DATA_ROOT` | `<repo>/our_data` |
| seturi de date TFRecord | `--datasets_root` | `COALITION4_DATASETS_ROOT` | `<data_root>/datasets` |
| checkpoint-uri | `--model_dir` | `COALITION4_MODEL_DIR` | `<repo>/models` |

Ordinea de precedență: parametru > variabilă de mediu > valoare implicită. `datasets/` se
rezolvă separat de `data_root`, astfel încât seturile de date pot fi stocate pe alt disc
decât colecția de patch-uri, de ordinul terabytes — anterior, acest lucru era posibil doar
printr-o joncțiune NTFS. **CRITIC** Blocajele de arhivare și marcajele de utilizare urmează
`--datasets_root`, menținând ciclul de arhivare și restaurare consecvent cu antrenarea.

### Recuperarea spațiului: comprimarea depozitelor `.npy`

Matricele reprezintă cea mai mare parte a proiectului — **5.292 GB în 1,10 milioane de
fișiere**, față de ~66 GB de seturi de date construite. Acestea sunt comprimate **direct pe
loc** cu zstd (`foo.npy` → `foo.npy.zst`), nu arhivate integral, deoarece fluxul de
procesare le deschide câte un cadru pe rând, după denumire. Fiecare cititor rezolvă
denumirea logică `.npy` către forma existentă pe disc, astfel încât **nu este necesară nicio
restaurare înaintea unei execuții**.

```bash
python compress_datasets.py --npy-stats our_data/reprojected_data     # estimare prealabilă
python compress_datasets.py --compress-npy our_data/reprojected_data our_data/patches
python compress_datasets.py --restore-npy DIR                         # și revenirea
```

| Destinație | Pe disc | Raport | După |
|---|---|---|---|
| `our_data/patches/` | 66,8 GB | 11,3× | 5,9 GB |
| `reprojected_data/satellite_data/MTG/` | 1313,4 GB | 8,8× | 148,5 GB |
| `reprojected_data/opera_data/` | 519,3 GB | 37,6× | 13,8 GB |
| `reprojected_data/lightning_data/` | 314,2 GB | 7110× | ~0 GB |
| `our_data/lightning_data/` | 313,4 GB | 16180× | ~0 GB |
| Depozitul MTG (E: + G:) | 2765,2 GB | 6,1× | 454,0 GB |
| **Total** | **5292,3 GB** | **8,5×** | **622,3 GB** |

- **Nivelul 10 aplicat pe `float32` exact în forma stocată** — fără schimbarea tipului de
  date, fără recuantizare, fără reordonarea octeților. Baza numerică rămâne neatinsă, astfel
  încât niciun calcul ulterior nu se poate modifica.
- **Siguranța la ștergere** — fiecare fișier este scris într-un `.tmp`, recitit de pe disc și
  comparat octet cu octet cu originalul înainte ca ceva să fie eliminat. Este stocat
  fișierul `.npy` integral, inclusiv antetul, astfel încât o restaurare este identică la
  nivel de octet prin construcție. **CRITIC**
- **Nu sunt arhive solide** — măsurat pe 24 de cadre consecutive, un flux solid unic aduce
  un câștig de 3 % (11,3× → 11,7×) și elimină accesul aleatoriu; o diferență temporală între
  cadre este chiar mai slabă (10,8×).
- `romania_grid_lats.npy` / `_lons.npy` nu sunt niciodată comprimate — sunt mici și citite
  de aproape toate scripturile.

**Ordinea obligatorie de execuție.**

1. Statisticile înaintea setului de date care le utilizează.
2. Verificarea contaminării înainte ca oricare model să citească datele de test.
3. Sumarul de acoperire înaintea completării din Data Store — completarea solicită exact ciclurile enumerate în fișierul JSON al pașilor de timp lipsă.
4. Sumarul de acoperire înaintea ștergerii datelor brute — odată eliminate, `--scan raw` nu mai poate descrie arhiva, rămânând valide doar perspectivele `npy` și `reprojected`.
5. Decuparea patch-urilor înaintea construirii seturilor de date, precum și re-decuparea ori de câte ori `patch_index.csv` este reconstruit — un fișier de patch consemnează plăcile după poziție, astfel încât o listă de activitate modificată deplasează în mod silențios placa aflată pe fiecare poziție.
