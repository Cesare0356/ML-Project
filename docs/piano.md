# Ablation dei regolarizzatori per MLP su dati tabulari

## In breve

Il progetto è un ablation study del paper _Well-tuned Simple Nets Excel on Tabular Datasets_ (Kadra et al., NeurIPS 2021), secondo cui un MLP con un "cocktail" di regolarizzatori batte XGBoost sui dati tabulari. Su 4 dataset del paper misuriamo l'effetto di 5 regolarizzatori in tre modi: aggiunti uno alla volta, tutti insieme, e tolti uno alla volta dal cocktail. Come modelli di confronto usiamo XGBoost e la regressione logistica. Gli effetti si confrontano con quelli del paper; il leave-one-out è il contributo nuovo. I risultati sono presentati in un poster.

Le scelte di dettaglio che questo piano richiama (D-01, D-02, …) sono in `docs/decisioni.md`, ognuna con la sua fonte nel paper o nel codice degli autori.

## Motivazione

Il paper misura l'effetto di ogni regolarizzatore preso da solo (Tabella 11), ma lascia aperte due cose:

- **quanto conta ogni regolarizzatore dentro il cocktail:** il paper non ne toglie mai uno per vedere quanto si perde;
- **quanto sono affidabili le differenze:** il paper riporta un solo numero per cella, senza margine di errore. Noi usiamo la stessa divisione 60/20/20, ma con 10 seed e un intervallo di confidenza per ogni confronto.

## Cosa fa il paper e cosa facciamo noi

|                             | Paper (Kadra et al. 2021)                                                                    | Noi                                                                                    |
| --------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Obiettivo                   | mostrare che un MLP con un cocktail di regolarizzatori, scelto per ogni dataset, batte i metodi più forti sui dati tabulari | misurare quanto conta ciascun regolarizzatore, da solo e dentro il cocktail |
| Dataset                     | 40 dataset OpenML                                                                            | 4 di quei 40, con gli stessi task ID                                                   |
| Divisione                   | una split 60/20/20 con seed 11; il test è uguale per tutti i metodi, la validazione no       | test identico; validazione stratificata con seed 11, uguale per tutti i modelli        |
| Preprocessing               | MLP: mediana per le numeriche, categoria costante per i mancanti delle categoriche, one-hot, standardizzazione, tutto stimato sul training; XGBoost: codifica ordinale delle categoriche e imputazione con il valore più frequente | uguale; in più, per l'MLP, rimozione delle feature costanti |
| Rete                        | MLP di 9 livelli da 512 unità ("shaped ResNet"), inizializzazione di default di PyTorch      | MLP scritto da noi (Linear, batch norm, ReLU, dropout), inizializzazione di default di PyTorch; numero di livelli e unità scelti dalla ricerca |
| Ottimizzatore               | AdamW, learning rate fisso 1e-3; weight decay su tutti i parametri                           | SGD con momentum 0,9, learning rate scelto dalla ricerca; L2 come weight decay su tutti i parametri |
| Addestramento               | batch 128, cross-entropy pesata, cosine annealing con restart                                | uguale                                                                                 |
| Regolarizzatori             | 13; da soli sulla rete senza regolarizzazione (Tabella 11), oppure la ricerca sceglie quali attivare e con quali iperparametri | 5; aggiunti uno alla volta alla rete senza L2, tutti insieme (sempre accesi), tolti uno alla volta |
| Ricerca degli iperparametri | BOHB con un solo livello di fidelity, su 19 iperparametri (Appendice A)                      | griglia che si sposta, random search, rivalutazione delle 3 migliori con 5 seed        |
| Ripetizioni                 | un solo valore per cella nella Tabella 11                                                    | 10 seed per configurazione                                                             |
| Modelli di confronto        | XGBoost, altre reti per dati tabulari (TabNet, NODE), sistemi AutoML (Auto-Sklearn, AutoGluon) | XGBoost e regressione logistica                                                      |
| Analisi statistica          | test di rango sui 40 dataset (Wilcoxon, critical difference)                                 | intervallo di confidenza bootstrap per ogni confronto, dataset per dataset             |
| Leave-one-out               | non fatto                                                                                    | esperimenti 8-12                                                                       |

**Cosa teniamo uguale e perché.** Sono le scelte che rendono i nostri risultati confrontabili con quelli del paper:
- dataset, metrica, test set e seed;
- loss pesata, batch, scheduler;
- il riferimento degli effetti, cioè la rete senza regolarizzazione;
- l'implementazione di snapshot ensembles e SWA;
- la L2 applicata a tutti i parametri;
- il preprocessing dell'MLP e di XGBoost.

Dove il paper non dava i dettagli li abbiamo presi dal codice degli autori, con file e riga in `docs/decisioni.md`.

**Cosa cambiamo e perché:**
- **rete e ottimizzatore:** usiamo un MLP semplice scritto da noi, con la ricerca degli iperparametri chiesta dal corso invece della configurazione fissa del paper;
- **validazione:** è stratificata correttamente e uguale per tutti i modelli. Nel codice dell'MLP degli autori la stratificazione non funziona (D-07), ma questo non tocca il test;
- **piccole differenze dichiarate** (in `docs/decisioni.md`): un solo tasso di dropout per tutti i livelli, l'ordine standard dei livelli dell'MLP, la rimozione delle feature costanti;
- **ricerca degli iperparametri:** ogni configurazione si tara con la stessa procedura, così il confronto tra regolarizzatori è equo;
- **ripetizioni e intervalli:** con 4 dataset i test di rango non hanno potenza, quindi serve un margine di errore per ogni dataset;
- **leave-one-out:** è la domanda che il paper non affronta.

**Cosa ne segue.** È una replica concettuale, non una riproduzione esatta. I valori assoluti dell'MLP saranno quasi certamente diversi da quelli del paper. Gli effetti dei regolarizzatori possono tornare o no, ed è proprio la domanda D1. Se un effetto non torna, non si può dire se dipende dalla rete o dall'ottimizzatore diversi, oppure dal rumore del numero del paper (vedi Regola di lettura).

## Domande

Non partiamo da risultati attesi: ogni esito è un risultato. Prima degli esperimenti fissiamo solo come si leggono i risultati.

1. **D1 (replica):** aggiungendo un regolarizzatore alla volta, ritroviamo gli effetti della Tabella 11?
2. **D2:** il cocktail migliora la rete senza regolarizzazione? Il guadagno è simile su tutti i dataset?
3. **D3:** quanto si perde togliendo dal cocktail ciascun regolarizzatore?
4. **D4:** sui nostri dataset il cocktail regge il confronto con XGBoost, come sostiene il paper?

**Regola di lettura.** Ogni effetto ha un intervallo di confidenza al 95% dal bootstrap appaiato (vedi Protocollo). Si legge in due passi:

1. **L'effetto si distingue dallo zero?** Se l'intervallo esclude lo zero sì, e il segno dice la direzione. Se lo contiene, l'effetto non si distingue dal rumore. Non vuol dire "nessun effetto": vuol dire "non misurabile con questi dati".
2. **Solo per D1, D2 e D4, il confronto con il paper (per D3 il paper non ha un valore):** il valore del paper cade dentro il nostro intervallo? Il paper dà un solo numero per cella, senza errore. Lo usiamo come punto di riferimento e non gli attribuiamo nessun errore, perché sarebbe un'ipotesi.

| Nostro intervallo | Valore del paper | Esito |
| --- | --- | --- |
| esclude lo 0, stesso segno del paper | dentro | replica: stessa direzione e ampiezza compatibile |
| esclude lo 0, stesso segno | fuori | stessa direzione, ampiezza diversa |
| esclude lo 0, segno opposto | — | direzione opposta |
| contiene lo 0 | dentro | non informativo: né replica né non replica |
| contiene lo 0 | fuori | non compatibile con l'effetto del paper |

Negli esiti "ampiezza diversa" e "non compatibile" restano aperte due spiegazioni: un effetto davvero diverso nel nostro setup, oppure il rumore del numero del paper, ottenuto con un solo seed. Non si sceglie tra le due: si riportano entrambe.

## Dataset

Quattro dei 40 dataset del paper, scaricati da OpenML con il task ID della Tabella 9 del paper.

| Dataset       | Task ID | Istanze | Feature | Classi |
| ------------- | ------- | ------- | ------- | ------ |
| credit-g      | 233088  | 1000    | 20      | 2      |
| phoneme       | 233103  | 5404    | 5       | 2      |
| mfeat-factors | 233093  | 2000    | 216     | 10     |
| cnae-9        | 233108  | 1080    | 856     | 9      |

**Divisione:** una sola, in training (60%), validazione (20%) e test (20%), con il seed 11, come nel paper (D-07).
- **Test:** si usa la stessa funzione degli autori, sui dati di OpenML nello stesso ordine: `train_test_split(X, y, test_size=0.2, random_state=11, stratify=y, shuffle=True)`.
- **Validazione:** `train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=11, stratify=y_trainval)`, uguale per tutti i modelli. È la divisione che gli autori usano per XGBoost. Per l'MLP gli autori ne usano un'altra, la cui stratificazione non funziona.
- **Controlli:** dimensioni e proporzioni delle classi di ogni parte. Gli esempi non si possono confrontare con quelli degli autori, che non pubblicano gli indici.

**Preprocessing**, stimato sui dati di addestramento: il training in model selection, training + validazione nel riaddestramento finale (D-02).
- **MLP e regressione logistica, come gli autori (D-12):**
  - feature numeriche: imputazione con la mediana e standardizzazione;
  - feature categoriche: i valori mancanti diventano una categoria a parte, poi one-hot;
  - in più rimuoviamo le feature costanti sul training.
- **XGBoost, come dice il testo del paper (D-10):** codifica ordinale delle feature categoriche invece del one-hot, imputazione con il valore più frequente. Il codice degli autori fa un po' diversamente, ma seguiamo il testo.

## Modelli

### MLP

Rete neurale feed-forward scritta da noi in PyTorch (D-08):
- ogni livello nascosto è Linear → batch norm (se attiva) → ReLU → dropout (se attivo);
- l'uscita è un Linear, con la softmax dentro la loss;
- l'inizializzazione è quella di default di PyTorch, come nel codice degli autori.

- **Addestramento:** SGD con momentum 0,9, batch 128, 105 epoche, senza early stopping. La loss è la cross-entropy pesata: la classe k pesa n / (K · n_k), calcolato sui dati di addestramento, come nel codice degli autori.
- **Learning rate:** segue un cosine annealing con restart. Scende da η a quasi zero lungo mezza curva coseno, poi riparte da η per un ciclo lungo il doppio: cicli di 15, 30 e 60 epoche. Serve agli snapshot ensembles, e lo usano tutte le configurazioni.
- **L2:** `weight_decay=λ` in SGD, su tutti i parametri, bias compresi, come il weight decay del paper. Con SGD equivale ad aggiungere λ/2·‖θ‖² alla loss. Il range è quello del paper, [1e-5; 0,1]. Negli esperimenti 1b e 2-6 vale λ = 0; nell'esperimento 1 λ è cercato nel range; nel cocktail e nel leave-one-out si cerca anche lo 0 (vedi Esperimenti).
- **Iperparametri cercati:** numero di livelli, unità per livello, learning rate η, più λ dove previsto e gli iperparametri dei regolarizzatori.

### Regolarizzatori

Cinque dei 13 del paper.

| Regolarizzatore                   | Cosa fa                                                    | Iperparametro cercato |
| --------------------------------- | ---------------------------------------------------------- | --------------------- |
| Dropout (DO)                      | spegne neuroni a caso durante il training                  | drop rate in (0; 0,8], uguale per tutti i livelli |
| Batch normalization (BN)          | normalizza le attivazioni di ogni livello                  | nessuno               |
| Snapshot ensembles (SE)           | media le softmax dei 3 modelli di fine ciclo (epoche 15, 45, 105) | nessuno        |
| Stochastic weight averaging (SWA) | media i pesi di fine ciclo (epoche 15, 45, 105), poi ricalcola le statistiche della batch norm sul training | nessuno |
| Mix-Up (MU)                       | addestra su combinazioni di coppie di esempi ed etichette, con coefficiente estratto da Beta(α, α) per ogni batch | α in (0; 1] |

Questi dettagli vengono dal codice degli autori (D-03, D-04):
- i range sono quelli del paper ma senza lo 0, perché nel paper lo 0 corrisponde a "spento" e l'attivazione è un flag a parte;
- con SE e SWA insieme, il terzo snapshot è il modello SWA;
- la loss di Mix-Up è λ·CE(y_a) + (1−λ)·CE(y_b), dove CE è la cross-entropy pesata e λ è il coefficiente estratto da Beta(α, α).

### Modelli di confronto

| Modello               | Perché                                                     | Iperparametri cercati                                                  |
| --------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------- |
| Regressione logistica | il modello lineare: dice quanto serve una rete non lineare | forza della L2                                                         |
| XGBoost               | il modello che il paper dice di battere                    | numero di alberi, profondità, learning rate, subsampling, penalità L2  |

Usano la stessa divisione e la stessa procedura di ricerca. Il preprocessing di XGBoost segue il paper (vedi Dataset). Girano su CPU in pochi minuti. La regressione logistica non dipende dal seed, quindi per lei basta un addestramento per combinazione.

**Pesi di classe (D-11):**
- **regressione logistica:** pesi bilanciati, come la loss pesata dell'MLP;
- **XGBoost:** nessun peso, come nel codice degli autori, così il confronto di D4 resta confrontabile con la Tabella 2.

**I valori delle griglie non sono ancora fissati:** si scelgono nella pilota, guardando solo training e validazione, e si congelano prima degli esperimenti.

## Esperimenti

| #  | Configurazione                | Domanda                                                     |
| -- | ----------------------------- | ----------------------------------------------------------- |
| 1  | Rete + L2 (λ nel range del paper) | colonna WD del paper: quanto vale la L2 da sola?        |
| 1b | Rete senza regolarizzazione (λ = 0) | colonna PN del paper: il riferimento degli esperimenti 1-7 |
| 2  | 1b + Dropout                  | quanto vale il dropout da solo?                             |
| 3  | 1b + Batch norm               | quanto vale la batch norm da sola?                          |
| 4  | 1b + Snapshot ensembles       | quanto valgono gli snapshot ensembles da soli?              |
| 5  | 1b + SWA                      | quanto vale SWA da solo?                                    |
| 6  | 1b + Mix-Up                   | quanto vale Mix-Up da solo?                                 |
| 7  | Cocktail (tutti e 5, con λ cercato, anche 0) | quanto vale il cocktail?                     |
| 8  | Cocktail − Dropout            | quanto si perde togliendo il dropout?                       |
| 9  | Cocktail − Batch norm         | quanto si perde togliendo la batch norm?                    |
| 10 | Cocktail − Snapshot ensembles | quanto si perde togliendo gli snapshot ensembles?           |
| 11 | Cocktail − SWA                | quanto si perde togliendo SWA?                              |
| 12 | Cocktail − Mix-Up             | quanto si perde togliendo Mix-Up?                           |
| 13 | Regressione logistica         | confronto lineare                                           |
| 14 | XGBoost                       | confronto con il modello che il paper dice di battere      |

Gli esperimenti 1, 1b e 2-6 rispondono a D1, il 7 a D2, gli esperimenti 8-12 a D3, il 13 e il 14 a D4. Come nel paper, ogni regolarizzatore "da solo" è aggiunto alla rete senza nessun'altra regolarizzazione, nemmeno la L2 (D-01).

## Protocollo

### Model selection (solo training e validazione)

Per ogni configurazione e dataset:

1. **Griglia di partenza:** ogni combinazione si addestra sul training e si valuta sulla validazione (balanced accuracy).
2. **La griglia si sposta:** se il valore vincente di un iperparametro è al bordo, si aggiunge il valore successivo in quella direzione e si provano solo le combinazioni nuove. Si ripete al massimo 2 volte per direzione.
3. **Random search fine:** 20 combinazioni nuove estratte intorno alla vincente, con il vicinato e il seed definiti in D-05.
4. **Controllo della fortuna:** le 3 combinazioni migliori si riaddestrano con altri 5 seed, e vince quella con la media più alta sui suoi 6 addestramenti.

### Model assessment (il test si usa una volta sola)

5. La combinazione vincente si riaddestra su training + validazione con 10 seed (D-09). Preprocessing e pesi della loss si ristimano su training + validazione (D-02).
6. Si valuta sul test. Il risultato è la media dei 10 seed ± la deviazione standard.

### Confronti

- **Con chi si confronta:** gli esperimenti 1 e 2-7 si confrontano con 1b, gli 8-12 con il cocktail, il 13 e il 14 con il cocktail.
- **Intervallo al 95%:** bootstrap appaiato sul test. Si ricampionano gli esempi di test 2.000 volte; ogni volta si ricalcola la differenza di balanced accuracy, mediata sui seed, usando gli stessi esempi per le due configurazioni. L'intervallo va dal 2,5° al 97,5° percentile.
- **Assunzione dichiarata:** il bootstrap tratta gli esempi di test come estratti in modo indipendente dalla stessa popolazione.
- **Prima degli esperimenti:** domande, griglie e codice di analisi si congelano con un commit. Il test e i numeri del paper non si usano per cambiare nulla.

## Confronto con il paper

Rete, ottimizzatore e tuning sono diversi da quelli del paper (vedi Cosa fa il paper e cosa facciamo noi), quindi i valori assoluti dell'MLP non devono coincidere. Si confrontano gli **effetti**: quanti punti si guadagnano o si perdono aggiungendo un regolarizzatore alla rete senza regolarizzazione.

Effetti nel paper, cioè differenza dalla colonna PN (la rete senza regolarizzazione) della Tabella 11, in punti di balanced accuracy:

| Dataset       | WD    | DO    | SE    | BN    | SWA    | MU    | Cocktail |
| ------------- | ----- | ----- | ----- | ----- | ------ | ----- | -------- |
| credit-g      | −0,11 | −1,30 | +0,36 | −0,71 | 0,00   | +0,60 | +5,24    |
| phoneme       | −0,77 | +0,88 | −0,48 | −0,49 | −0,29  | −0,39 | −0,89    |
| mfeat-factors | +0,25 | +0,25 | 0,00  | +0,75 | −28,50 | +0,25 | +0,25    |
| cnae-9        | +4,17 | +3,24 | +0,46 | +3,70 | −37,50 | +6,94 | +8,33    |

I valori sono stati ricalcolati dalle Tabelle 11 e 2 del paper e coincidono con quelli riportati qui.

**Scala degli effetti (D-06).** Il paper non dà errori, e noi non gliene attribuiamo. Per dare un'idea di quanto sono grandi i suoi effetti, si dice a quanti esempi di test corrispondono.
- Un esempio della classe k che passa da sbagliato a giusto cambia la balanced accuracy di 100 / (K · n_k) punti, dove n_k è il numero di esempi di test della classe k.
- Per esempio, su mfeat-factors (10 classi, 40 esempi di test per classe) un esempio vale 0,25 punti: l'effetto +0,25 corrisponde a un solo esempio.
- I valori per i quattro dataset si calcolano dai dati al download.

Per **D4** il paper riporta nella Tabella 2 XGBoost e il cocktail sugli stessi dataset:

| Dataset       | XGBoost | Cocktail | Differenza |
| ------------- | ------- | -------- | ---------- |
| credit-g      | 68,93   | 74,64    | +5,71      |
| phoneme       | 87,97   | 86,62    | −1,35      |
| mfeat-factors | 98,00   | 98,00    | 0,00       |
| cnae-9        | 94,91   | 95,83    | +0,93      |

**Come si confronta un effetto:** con i due passi della Regola di lettura (vedi Domande): prima il segno e l'intervallo del nostro effetto, poi se il valore del paper cade dentro l'intervallo.

Il paper non misura il leave-one-out: lì l'unico riferimento è la Figura 3, cioè quanto spesso la ricerca del paper sceglie ciascun regolarizzatore.

## Grafici

**Regole:**
- dove il paper ha un dato corrispondente, sta nello stesso grafico: in grigio il paper, a colori noi;
- sullo stesso asse vanno gli effetti, non i valori assoluti;
- i numeri del paper si ridisegnano citando la fonte, senza incollare le sue figure.

**I grafici:**
1. **Tabella principale:** balanced accuracy di test (media ± deviazione standard) per configurazione e dataset, con accanto i valori del paper (Tabelle 11 e 2).
2. **Curve di addestramento** della rete senza regolarizzazione (1b) e del cocktail: loss e balanced accuracy per epoca su training e validazione, media ± deviazione standard tra seed, restart del learning rate segnati. Mostrano convergenza e overfitting.
3. **Effetti dei singoli regolarizzatori:** per ogni dataset, barra del paper e barra nostra affiancate. Solo la nostra barra ha l'intervallo al 95%, perché il paper non dà errori.
4. **Cocktail e leave-one-out:** guadagno del cocktail (paper e noi), poi la perdita togliendo ciascun regolarizzatore, con accanto la frequenza della Figura 3.
5. **Cocktail contro XGBoost:** per ogni dataset, la differenza tra i due nel paper e da noi.
6. **Sintesi:** effetto del paper sull'asse x, effetto nostro sull'asse y, con la diagonale y = x.

## Tempi

Il numero di addestramenti dipende dalle griglie, che si fissano nella pilota. Per ogni configurazione e dataset servono:
- le combinazioni della griglia, più le eventuali estensioni;
- 20 combinazioni di random search;
- 15 addestramenti di rivalutazione (3 combinazioni × 5 seed);
- 10 addestramenti finali.

Per dare un ordine di grandezza: con una griglia di circa 50 combinazioni per la rete senza regolarizzazione sarebbero circa 8.400 addestramenti dell'MLP, poco più di 2 ore su una T4. Lo script `benchmark_tempi.py` misura il tempo reale per addestramento e calcola il totale dalle dimensioni delle griglie.

**Su Colab:** dati già sulla GPU, senza DataLoader; sessioni di 2-3 ore, con ogni risultato salvato su Drive appena finisce.

## Ordine di lavoro

1. **Pilota**, solo su training e validazione:
   - verifica che i regolarizzatori funzionino;
   - scelta dei valori delle griglie e dei seed (D-09);
   - misura dei tempi.
2. **Congelamento** di domande, griglie e codice con un commit.
3. **Esperimenti 1 e 1b**, con controlli che non dipendono dal risultato: la loss scende, la rete fa meglio di un classificatore a caso, nessun dato di validazione o test entra nel training.
4. **Esperimenti 2-14.**
5. **Confronto con il paper, grafici, poster.**

## Limiti

- **Una sola divisione, come nel paper:** con un'altra divisione i risultati potrebbero cambiare, e il bootstrap misura solo il rumore del test.
- **Il paper non riporta errori:** quando un nostro effetto ha un'ampiezza diversa da quella del paper, non si distingue tra un effetto diverso nel nostro setup e il rumore del numero del paper.
- **Assunzione del bootstrap:** gli esempi di test sono trattati come estratti in modo indipendente dalla stessa popolazione.
- **Esempi di test non verificabili:** usiamo la stessa funzione e lo stesso seed degli autori, ma non possiamo confrontare gli esempi con i loro, perché non pubblicano gli indici.
- **Setup diverso dal paper:** si confrontano gli effetti, non i numeri; il cocktail ha 5 regolarizzatori invece di 13.
- **Ricerca sulla stessa validazione:** una combinazione può vincere per fortuna. La rivalutazione con 5 seed lo limita.
- **Molti confronti:** qualche differenza può risultare significativa per caso.
- **Generalità limitata:** 4 dataset, solo classificazione.

## Riferimenti

Tutti i riferimenti sono pubblicati con revisione tra pari; il link arXiv del paper principale è solo un accesso comodo al testo.

- Kadra, A., Lindauer, M., Hutter, F., Grabocka, J. _Well-tuned Simple Nets Excel on Tabular Datasets_. Advances in Neural Information Processing Systems 34 (NeurIPS 2021). Tabelle 1, 2, 4, 9 e 11, Figura 3, Appendici A, B.1 e C.1. Testo: [arXiv:2106.11189](https://arxiv.org/abs/2106.11189). Codice: [WellTunedSimpleNets](https://github.com/releaunifreiburg/WellTunedSimpleNets) (commit `5405846`) e il branch [`regularization_cocktails` di Auto-PyTorch](https://github.com/automl/Auto-PyTorch/tree/regularization_cocktails) (commit `ce6e575`), dove sono implementati i regolarizzatori.
- Chen, T., Guestrin, C. _XGBoost: A Scalable Tree Boosting System_. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016).
- Srivastava, N. et al. _Dropout: A Simple Way to Prevent Neural Networks from Overfitting_. Journal of Machine Learning Research 15, 2014.
- Ioffe, S., Szegedy, C. _Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift_. International Conference on Machine Learning (ICML 2015).
- Loshchilov, I., Hutter, F. _SGDR: Stochastic Gradient Descent with Warm Restarts_. International Conference on Learning Representations (ICLR 2017).
- Huang, G. et al. _Snapshot Ensembles: Train 1, Get M for Free_. International Conference on Learning Representations (ICLR 2017).
- Izmailov, P. et al. _Averaging Weights Leads to Wider Optima and Better Generalization_. Conference on Uncertainty in Artificial Intelligence (UAI 2018).
- Zhang, H. et al. _mixup: Beyond Empirical Risk Minimization_. International Conference on Learning Representations (ICLR 2018).
- Bergstra, J., Bengio, Y. _Random Search for Hyper-Parameter Optimization_. Journal of Machine Learning Research 13, 2012.
- Efron, B., Tibshirani, R. _An Introduction to the Bootstrap_. Chapman & Hall, 1993.