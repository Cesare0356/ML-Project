# Registro delle decisioni

Qui finisce ogni scelta che `docs/piano.md` non fissava, con la sua fonte. Il piano è già aggiornato con le decisioni prese il 2026-11-25.

Formato di ogni voce nuova:

```
## D-NN Titolo
- Stato: aperta | decisa il AAAA-MM-GG
- Problema: cosa il piano non dice o dice in modo ambiguo
- Fonti: paper (sezione/tabella) e codice (file:riga)
- Decisione:
- Rispetto al paper: fedele | deviazione (motivo)
```

## Fonti

- **[P]** Kadra et al. 2021, _Well-tuned Simple Nets Excel on Tabular Datasets_, arXiv:2106.11189v2.
- **[W]** `releaunifreiburg/WellTunedSimpleNets`, commit `5405846`: split, configurazione dei run, baseline.
- **[A]** `automl/Auto-PyTorch`, branch `regularization_cocktails`, commit `ce6e575`: implementazione dei regolarizzatori. I percorsi partono da `autoPyTorch/`.

---

## D-01 Rispetto a cosa si misurano gli effetti dei singoli regolarizzatori
- Stato: decisa il 2026-11-25
- Problema: il piano confrontava gli esperimenti 2-6 con una baseline che ha la L2 scelta dalla ricerca, mentre il paper li confronta con la rete senza regolarizzazione.
- Fonti:
  - [P] Appendice C.1: ogni colonna della Tabella 11 è la rete "regolarizzata con una sola tecnica alla volta", e WD è una delle colonne; "PN" è la rete base.
  - [W] `utilities.py:333-402`: ogni regolarizzatore ha il suo flag, e il weight decay è spento se non lo si chiede (`utilities.py:363-368`).
  - [A] `pipeline/components/setup/optimizer/AdamWOptimizer.py:98`: range del weight decay (1e-5, 0,1).
- Decisione:
  - 1b (λ = 0) è la rete senza regolarizzazione (PN) ed è il riferimento degli esperimenti 1 e 2-7.
  - L'esperimento 1 è "rete + L2", con λ in [1e-5; 0,1]: corrisponde alla colonna WD.
  - Gli esperimenti 2-6 si fanno con λ = 0.
  - Nel cocktail la L2 è cercata, con lo 0 incluso, come nel paper, dove il weight decay è uno degli ingredienti che la ricerca può spegnere.
  - Il leave-one-out (8-12) si confronta con il cocktail.
- Rispetto al paper: fedele.

## D-02 Preprocessing nel riaddestramento finale
- Stato: decisa il 2026-11-25
- Fonti:
  - [P] Sezioni 4.1 e 5.1: la configurazione scelta si riaddestra su X^(Train) ∪ X^(Val).
  - [W] `cocktails/main_experiment.py:353-361` e `384-392`: il riaddestramento usa tutto l'80% senza divisione, e la pipeline stima lì il preprocessing.
- Decisione: nel riaddestramento finale (passo 5) il preprocessing si ristima su training + validazione. Il test resta fuori.
- Rispetto al paper: fedele.

## D-03 Snapshot ensembles e SWA
- Stato: decisa il 2026-11-25
- Problema: il paper non dice quando si prendono gli snapshot, né quando e come SWA media i pesi.
- Fonti:
  - [A] `pipeline/components/training/trainer/base_trainer.py:312-331`: a fine ciclo SWA aggiorna la media e SE salva uno snapshot; si tengono gli ultimi 3; se c'è anche SWA, l'ultimo snapshot è il modello SWA.
  - [A] `pipeline/components/training/trainer/base_trainer_choice.py:403-412`: a fine training si ricalcolano le statistiche della BN sul training; la rete prende i pesi SWA.
  - [A] `pipeline/components/setup/network/base_network.py:64` e `114-125`: la predizione di SE è la media delle softmax degli snapshot.
  - [A] `pipeline/components/setup/lr_scheduler/CosineAnnealingWarmRestarts.py:55-57`: T_0 = 15 e T_mult = 2 con 105 epoche, cioè cicli di 15, 30 e 60.
- Decisione:
  - **SWA:** la media semplice dei pesi alla fine di ogni ciclo (epoche 15, 45, 105). Dopo la media si ricalcolano le statistiche della batch norm sul training, e la rete finale ha i pesi mediati.
  - **SE:** i 3 modelli di fine ciclo; la predizione è la media delle loro softmax.
  - **SE e SWA insieme:** gli snapshot sono il modello all'epoca 15, quello all'epoca 45 e il modello SWA, con la BN ricalcolata.
- Rispetto al paper: fedele al codice.

## D-04 Range dei regolarizzatori e lo 0
- Stato: decisa il 2026-11-25
- Fonti:
  - [P] Tabella 1: drop rate in [0; 0,8], intensità del Mix-Up in [0; 1], e ogni regolarizzatore ha un flag di attivazione separato dal suo valore.
  - [A] `pipeline/components/training/trainer/MixUpTrainer.py:33`: con α = 0 il coefficiente vale 1, cioè nessun mescolamento.
  - [A] `pipeline/components/setup/network_backbone/ShapedResNetBackbone.py:48-57` e `138-142`: il dropout ha un tasso massimo e una "forma" che lo distribuisce tra i livelli.
- Decisione:
  - Lo 0 vuol dire spento, quindi le griglie escludono lo 0: dropout in (0; 0,8], α di Mix-Up in (0; 1].
  - Il dropout ha un solo tasso, uguale per tutti i livelli.
- Rispetto al paper: fedele per i range. Il tasso unico è una deviazione: equivale alla forma "brick" del paper, ed è la scelta più semplice con un MLP generico.

## D-05 Random search intorno alla vincente
- Stato: decisa il 2026-11-25
- Problema: il piano non definiva il vicinato. Il paper usa BOHB, quindi lì non c'è un riferimento.
- Decisione: per ogni iperparametro, sia g_i il valore vincente della griglia, con vicini g_(i−1) e g_(i+1). Se g_i è al bordo, si usa solo il lato che esiste.
  - **η e λ:** estrazione log-uniforme tra i due vicini.
  - **Drop rate e α:** estrazione uniforme tra i due vicini, restando dentro i range di D-04.
  - **Livelli e unità:** uno dei tre valori g_(i−1), g_i, g_(i+1).
  - **Se il λ vincente è 0** (solo dove lo 0 è ammesso): λ resta 0 nella random search.
  - **Campionamento:** generatore con seed 11. Le combinazioni già provate si scartano e si estraggono di nuovo, finché se ne hanno 20 nuove.
- Rispetto al paper: non applicabile (procedura nostra).

## D-06 Come si leggono gli effetti e come si confrontano con il paper
- Stato: decisa il 2026-11-25
- Problema: il paper dà un solo numero per cella, senza errore. Qualunque errore attribuito al paper (formula binomiale, fattore √2) sarebbe un'ipotesi.
- Decisione:
  - Il bootstrap appaiato resta. Si ricampionano gli esempi di test 2.000 volte e si ricalcola la differenza di balanced accuracy mediata sui 10 seed, usando gli stessi esempi per le due configurazioni.
  - Si legge in due passi:
    1. l'intervallo al 95% esclude lo zero? Se sì, con quale segno?
    2. il valore del paper cade dentro l'intervallo?
  - L'esito si classifica con la tabella della "Regola di lettura" del piano.
  - Al paper non si attribuisce nessun errore. Per dare la scala degli effetti del paper, si dice a quanti esempi di test corrispondono: è aritmetica, non un'ipotesi.
- Assunzione dichiarata: il bootstrap tratta gli esempi di test come estratti in modo indipendente dalla stessa popolazione.
- Limite dichiarato: negli esiti "ampiezza diversa" e "non compatibile" non si distingue tra un effetto diverso nel nostro setup e il rumore del numero del paper.
- Rispetto al piano: sostituisce la formula 2·√(e.s.²_nostro + e.s.²_paper) e il paragrafo sull'errore dei numeri del paper.

## D-07 Split
- Stato: decisa il 2026-11-25
- Fonti:
  - [W] `utilities.py:71-78`: il test si estrae con `train_test_split(X, y, test_size=0.2, random_state=11, stratify=y, shuffle=True)`, dove X e y vengono da `task.get_dataset().get_data(target=dataset.default_target_attribute)`. `baselines/bohb/utilities.py:127-133` fa lo stesso per XGBoost.
  - [A] `datasets/base_dataset.py:127`, `217-218` e `317-322`, con `datasets/resampling_strategy.py:138-145`: per l'MLP la validazione si estrae da indici permutati con `RandomState(11)`, ma a `stratify` arrivano le etichette nell'ordine originale. Etichette e indici quindi non corrispondono, e la validazione di fatto non è stratificata.
  - [W] `baselines/bohb/utilities.py:146-154`: per XGBoost la validazione è una divisione stratificata corretta, con seed 11. Nel paper, quindi, l'MLP e XGBoost hanno validazioni diverse.
- Decisione:
  - **Test:** identico al paper, con la stessa funzione, lo stesso seed e lo stesso ordine dei dati di OpenML.
  - **Validazione:** `train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=11, stratify=y_trainval)`, come la validazione di XGBoost degli autori, uguale per tutti i modelli.
  - **Controlli:** dimensioni e proporzioni delle classi di ogni parte. Gli esempi non si possono confrontare con quelli degli autori, che non pubblicano gli indici.
- Rispetto al paper: fedele sul test. Sulla validazione è una deviazione, che corregge il difetto e non tocca il test.

## D-08 MLP e L2
- Stato: decisa il 2026-11-25
- Problema: il corso ha fatto solo teoria, quindi non c'è un codice di riferimento.
- Fonti:
  - [W] `utilities.py:180`: nessuna inizializzazione speciale, si usa quella di default di PyTorch.
  - [A] `pipeline/components/setup/optimizer/AdamWOptimizer.py:63-68`: il weight decay si applica a tutti i parametri della rete.
  - [A] `pipeline/components/setup/network_backbone/ResNetBackbone.py:300-319`: l'ordine dei livelli nel blocco del paper.
- Decisione:
  - **Livelli:** l'MLP è scritto da noi in PyTorch. Ogni livello nascosto è Linear → BatchNorm (se attiva) → ReLU → Dropout (se attivo), e l'uscita è un Linear con softmax nella loss.
  - **Inizializzazione:** quella di default di PyTorch.
  - **L2:** `weight_decay=λ` in `torch.optim.SGD`, su tutti i parametri, bias e parametri della batch norm compresi. Con SGD equivale ad aggiungere λ/2·‖θ‖² alla loss.
- Rispetto al paper:
  - fedele per l'inizializzazione e per i parametri a cui si applica la L2;
  - è una deviazione l'ordine dei livelli: il blocco del paper mette batch norm e ReLU prima del Linear, e il dropout solo a metà di ogni blocco da due livelli;
  - era già dichiarata nel piano la differenza tra SGD e AdamW.

## D-09 Valori dei seed
- Stato: aperta, da fissare prima del congelamento
- Problema: il piano dice "altri 5 seed" e "10 seed", ma non quali.
- Proposta da discutere:
  - model selection con il seed 11, come il paper;
  - rivalutazione con i seed 12-16;
  - riaddestramento finale con i seed 1-10;
  - tutti i seed registrati in ogni file di risultato.

---

## Altre differenze tra piano, paper e codice

## D-10 Preprocessing di XGBoost
- Stato: decisa il 2026-11-25
- Fonti:
  - [P] Appendice B.2: codifica ordinale delle categoriche, imputazione con il valore più frequente.
  - [W] `baselines/bohb/utilities.py:97-118`: il codice fa altro. Imputa con una costante, standardizza, e il one-hot è un iperparametro cercato (`use_ohe`).
- Decisione: si segue il testo del paper (ordinale e valore più frequente), come diceva già il piano. La discrepanza con il codice si dichiara.
- Nota: tra i quattro dataset solo credit-g ha feature categoriche. Se ci siano valori mancanti si verifica al download.

## D-11 Pesi di classe dei modelli di confronto
- Stato: decisa il 2026-11-25
- Fonti:
  - [P]: nessun peso di classe menzionato per XGBoost.
  - [W] `baselines/bohb/worker.py:137-139`: le `DMatrix` non hanno pesi.
- Decisione:
  - **XGBoost:** senza pesi di classe, come gli autori, così i numeri di D4 restano confrontabili.
  - **Regressione logistica:** pesi bilanciati, come la loss pesata dell'MLP. Non è nel paper, quindi non c'è un riferimento a cui essere fedeli.

## D-12 Preprocessing dell'MLP e della regressione logistica
- Stato: decisa il 2026-11-25
- Fonti:
  - [W] `utilities.py:290-316`: per le numeriche, imputazione con la mediana e standardizzazione; per le categoriche, valori mancanti come categoria costante e one-hot.
  - [A] la pipeline non rimuove le feature costanti.
- Decisione:
  - Lo stesso preprocessing degli autori: one-hot c'era già nel paper, non è un'aggiunta nostra.
  - In più si rimuovono le feature costanti sul training. Una colonna costante nel training ma diversa da zero nel test arriverebbe a pesi mai addestrati.
- Rispetto al paper: fedele. La rimozione delle feature costanti è una deviazione dichiarata.

## D-13 Pesi della loss dell'MLP
- Stato: verificata, nessun cambiamento
- Fonti:
  - [A] `utils/implementations.py:37-39`: il peso della classe k è n / (K · n_k).
  - [A] `pipeline/components/training/trainer/base_trainer_choice.py:338`: i pesi si calcolano sulle etichette del training.
- Decisione: si fa lo stesso. Peso della classe k = n / (K · n_k), calcolato sui dati con cui si addestra: il training in selezione, training + validazione nel riaddestramento finale.
