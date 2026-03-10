Struttura dataset:
## Informazioni Generali
- **Info**: Subset di sviluppo (development set) con speech pulito (clean), circa 5 ore totali.
- **Speaker**: ~40 parlanti unici (US English accent), audio disgiunto da altri subset.
- **Formato audio**: FLAC, 16kHz mono.


## Struttura della Cartella
La root del dataset contiene:
- File metadata: README.TXT, SPEAKERS.TXT (genere/durata speaker), CHAPTERS.TXT (durate capitoli), BOOKS.TXT (titoli libri).
- Subdirectory `dev-clean/` con la gerarchia:
  ```
  
  └── dev-clean/
      ├── <speaker_id>/          # Es. 84, 1272, 1358...
      │   ├── <chapter_id>/       # Es. 121123, 128104, 141231...
      │   │   ├── <speaker>-<chapter>.trans.txt   # Trascrizioni
      │   │   ├── <speaker>-<chapter>-0000.flac   # Audio utterance 1
      │   │   ├── <speaker>-<chapter>-0001.flac   # Audio utterance 2
      │   │   └── ...
      │   └── <altro_chapter>/
      └── <altro_speaker>/
          └── ...
  ```

## Formato File .trans.txt
```
84-121123-0000 PLEASE DO NOT THROW ANYTHING ON THE TRACK
84-121123-0001 WHETHER YOU ARE ON A TRAIN PLATFORM OR NOT
...
```
- Prima colonna: ID utterance (`speaker-chapter-id_sequenziale`).
- Seconda: trascrizione normalizzata.

## Note
- Audio segmentato su silenzi/pause.
