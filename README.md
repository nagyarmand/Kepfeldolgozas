# Forgalomszámláló alkalmazás 


### Probléma és motiváció
A közlekedési forgalom folyamatos monitorozása és elemzése kulcsfontosságú a hatékony közlekedésszervezéshez és balesetmegelőzéshez. A manuális adatgyűjtési módszerek lassúak és pontatlanok, különösen nagy forgalmú környezetben. Az automatizált rendszerek képesek valós időben nyomon követni a járműveket, és pontos adatokat biztosítani a közlekedési dinamikáról.

### A Projekt célkitűzései
A projekt célja egy olyan rendszer fejlesztése, amely:
- Valós idejű járműdetektálást végez különböző algoritmusokkal.
- Azonosítja és követi a járműveket a videókban.
- Számlálja a meghatározott áthaladási vonalakon átkelő járműveket.
- Kétféle detektor és tracker megvalósítása.


## Megoldáshoz szükséges elméleti háttér

#### Képfeldolgozás
A képfeldolgozás a digitális képek elemzésének tudománya, amely során a képek feldolgozására, szűrésére és előfeldolgozására különböző algoritmusokat alkalmazunk. Az alkalmazott eljárások közé tartozik:

#### Grey-scale konverzió
A képek szürkeárnyalatossá alakítása csökkenti a számítási költségeket, miközben megőrzi a szükséges információkat.

#### Zajszűrés
A zajszűrés a képfeldolgozás pontosságának növelésére szolgál. Gaussian blur-t használunk, hogy eltávolítsuk a zajokat és simítsuk a képet.

#### Háttérszubtrakció
A statikus háttér eltávolításával a mozgó objektumok könnyebben detektálhatók. Az OpenCV `createBackgroundSubtractorMOG2` függvénye árnyékok kezelésére is alkalmas.

#### Morphológiai nűveletek
A morfológiai műveletek segítenek a zaj csökkentésében, az objektumok kontúrjainak javításában és az egyes részek kiemelésében vagy eltüntetésében. Ezek a műveletek bináris képeken alkalmazhatók, ahol a képpontok értékei általában 0 vagy 1 (fekete és fehér).
- **Erozió:** Az erózió egy objektum kontúrjait zsugorítja. Ez különösen hasznos a kisebb zajok eltávolítására vagy az objektumok közötti keskeny kapcsolatok megszüntetésére. Az OpenCV `cv2.erode` függvényét használjuk erre a célra.
- **Dilatáció:** A dilatáció az objektum kontúrjait kibővíti. Ez a módszer segít a kis szakadások kitöltésében, valamint a háttérből kinyúló objektumok összekapcsolásában. Az OpenCV `cv2.dilate` függvénye biztosítja a dilatációt.
- **Nyitás (Opening):** A nyitás az erózió és dilatáció egymást követő alkalmazása, amely először eltávolítja a kisebb zajokat (erózióval), majd visszaállítja az eredeti objektum méretét (dilatációval). A nyitást gyakran használják kis fehér zaj eltávolítására.
- **Zárás (Closing):** A zárás a dilatáció és erózió egymást követő alkalmazása, amely először kibővíti az objektumokat (dilatációval), majd visszaállítja azok eredeti méretét (erózióval). Ez a módszer hasznos a kis fekete zaj eltüntetésére vagy az objektumok közötti kis hézagok kitöltésére.

#### Kontúrok keresése
A `cv2.findContours` függvény az objektumok körvonalait detektálja, amelyekből bounding boxokat generálunk.

##
### Objektumdetektálás

#### Custom detektor
- **Módszer**: Háttérszubtrakció és kontúrkeresés.
- **Előnyök**: Egyszerű, gyors, kis számítási igény.
- **Hátrányok**: Pontatlan zsúfolt vagy változó környezetben.

#### YOLO detektor
- **Módszer**: Gépi tanuláson alapuló járműdetektálás.
- **Előnyök**: Pontos, képes a különböző objektumtípusok azonosítására.
- **Hátrányok**: Nagyobb számítási igény.
  
##
### Objektumkövetés

#### Euclidean tracker
- Az objektumok középpontjai közötti Euklideszi távolságot használja.
- Egyszerű és gyors, de zsúfolt környezetekben kevésbé pontos.

#### SORT (Simple Online and Realtime Tracking)
- Kalman-szűrőt és az IoU metrikát használja az objektumok pontos követésére.
- Pontosabb, mint az Euclidean Tracker, de nagyobb gépigényű.

## A megvalósítás terve és kivitelezése

### Rendszer felépítése
**A rendszer fő komponensei:**
- **Detektorok** (Custom és YOLO): Az objektumok azonosításáért felelnek.
- **Tracker-ek** (SORT és Eukledian): Az objektumok mozgásának nyomon követése és azonosítása.
- **Felhasználói interfész** (Application): A teljes folyamat vezérlése, vizualizáció.

**Bemenetek:**
- **Videók:**
    - media/video1.mp4,
    - media/video2.mp4,
    - media/video3.mp4,
    - media/video4.mp4,
    - media/video4.mp4.
- **Maszkok:**
  - media/mask1.png,
  - media/mask2.png,
  - media/mask3.png,
  - media/mask4.png,
  - media/mask5.png,
##
**Adatfeldolgozás:**
1.	**Videók beolvasása:** A videók képkockánként kerülnek feldolgozásra.
2.	**Detektálás:** A YOLO vagy háttérkivonásos módszer használata az objektumok azonosítására.
3.	**Követés:** Az azonosított objektumokat az IoU (SORT) vagy az Euklideszi távolság (Euclidean) segítségével követjük.
4.	**Számlálás:** Ha egy objektum középpontja áthalad egy vonalon, a számláló növekszik.

### Rendszerarchitektúra
Az alábbi folyamatábra szemlélteti a rendszer működését:
- Videó → Detektor (Custom/YOLO) → Tracker (SORT/Euclidean) → Számlálás → Kimenet (Vizualizáció)

##
### Osztályok és funkciók
#### YOLODetector
- Importálja a YOLO modellt.
- A bounding box koordinátáit adja vissza.
   Csak autókat, buszokat, motorokat és teherautókat detektál.
#### CustomDetector
- Háttérkivonást és morfológiai műveleteket alkalmaz.
- Egyszerű, tanításmentes megoldás a mozgó objektumok felismerésére.
#### Tracker
- Euklédeszi alapokon készült, egyszerű távolság alapú tracker.
#### VideoProcessor
- Egységesíti a detektálási, követési és számlálási folyamatot.
- A középpontokat és a bounding box-okat megjeleníti a képkockákon.
#### Application
- Vezérli a feldolgozást és biztosítja a felhasználói interakciókat (pl. videók közötti váltás).


## Tesztelés

### CUDA tesztelése
A rendszerhez tartozik egy test.py nevű fájl, amely a CUDA-kompatibilis GPU elérhetőségének tesztelésére szolgál.

A kimenet megmutatja, hogy a rendszer képes-e CUDA gyorsítást használni:
- Ha CUDA elérhető, megjeleníti a GPU nevét.
- Ha nincs elérhető GPU, "Nincs GPU" üzenet jelenik meg.
##
### Számlálás tesztelése
#### Első videó
- Valós érték:
- Számlált járművek száma:
  - YOLO + SORT:
  - YOLO + Euclidean:
  - Custom + SORT:
  - Custom + Euclidean:

#### Második videó
- Valós érték:
- Számlált járművek száma:
  - YOLO + SORT:
  - YOLO + Euclidean:
  - Custom + SORT:
  - Custom + Euclidean:

#### Harmadik videó
- Valós érték:
- Számlált járművek száma:
  - YOLO + SORT:
  - YOLO + Euclidean:
  - Custom + SORT:
  - Custom + Euclidean:

#### Negyedik videó
- Valós érték:
- Számlált járművek száma:
  - YOLO + SORT:
  - YOLO + Euclidean:
  - Custom + SORT:
  - Custom + Euclidean:

#### Ötödik videó
- Valós érték:
- Számlált járművek száma:
  - YOLO + SORT:
  - YOLO + Euclidean:
  - Custom + SORT:
  - Custom + Euclidean:
##
### Összegzés
(teszteredmények összehasonlítása)


## Felhasználói útmutató
1.  CUDA és cuDNN telepítése
2.	Telepítse a szükséges csomagokat: **pip install -r requirements.txt**
3.	CUDA tesztelése a **test.py** használatával.
4.	Töltse le a médiafájlokat a következő Google Drive link használatával, és helyezze őket a /media mappába: (https://drive.google.com/drive/folders/1nm1JKHnu46o-tzZtBkBxpUDp9bs3cQuw?usp=drive_link)
5.	Futtassa a programot: **Teljes.py**

##
**A felhasználói vezérlés:**
-	**1:** az első videó feldolgozásának elindítása.
-	**2:** A második videó feldolgozásának elindítása.
-	**3:** A harmadik videó feldolgozásának elindítása.
-	**4:** A negyedik videó feldolgozásának elindítása.
- **5:** Az ötödik videó feldolgozásának elindítása.	
-	**ESC:** Kilépés az alkalmazásból.

## Összegzés


## Irodalomjegyzék
1.	Redmon, J., Farhadi, A. (2018). YOLOv3: An Incremental Improvement.
2.	Bewley, A., Ge, Z., et al. (2016). Simple Online and Realtime Tracking.
3.	OpenCV Documentation - https://opencv.org
