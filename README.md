# Forgalomszámláló alkalmazás 

### 1. Probléma és motiváció
A közlekedési forgalom folyamatos monitorozása és elemzése kulcsfontosságú a hatékony közlekedésszervezéshez és balesetmegelőzéshez. A manuális adatgyűjtési módszerek lassúak és pontatlanok, különösen nagy forgalmú környezetben. Az automatizált rendszerek képesek valós időben nyomon követni a járműveket, és pontos adatokat biztosítani a közlekedési dinamikáról.

### 1.1. A Projekt célkitűzései
A projekt célja egy olyan rendszer fejlesztése, amely:
- Valós idejű járműdetektálást végez különböző algoritmusokkal.
- Azonosítja és követi a járműveket a videókban.
- Számlálja a meghatározott áthaladási vonalakon átkelő járműveket.
- Többféle detektálási és követési módszert alkalmaz, például háttérszubtrakciót és mélytanulást.

## 2. Megoldáshoz szükséges elméleti háttér

### 2.1. Képfeldolgozás
A képfeldolgozás a digitális képek elemzésének tudománya, amely során a képek feldolgozására, szűrésére és előfeldolgozására különböző algoritmusokat alkalmazunk. Az alkalmazott eljárások közé tartozik:

#### 2.1.1. Grey-scale konverzió
A képek szürkeárnyalatossá alakítása csökkenti a számítási költségeket, miközben megőrzi a szükséges információkat.

#### 2.1.2. Zajszűrés
A zajszűrés a képfeldolgozás pontosságának növelésére szolgál. Gaussian blur-t használunk, hogy eltávolítsuk a zajokat és simítsuk a képet.

#### 2.1.3. Háttérszubtrakció
A statikus háttér eltávolításával a mozgó objektumok könnyebben detektálhatók. Az OpenCV `createBackgroundSubtractorMOG2` függvénye árnyékok kezelésére is alkalmas.

#### 2.1.4. Morphológiai nűveletek
A morfológiai műveletek segítenek a zaj csökkentésében, az objektumok kontúrjainak javításában és az egyes részek kiemelésében vagy eltüntetésében. Ezek a műveletek bináris képeken alkalmazhatók, ahol a képpontok értékei általában 0 vagy 1 (fekete és fehér).
- **Erozió:** Az erózió egy objektum kontúrjait zsugorítja. Ez különösen hasznos a kisebb zajok eltávolítására vagy az objektumok közötti keskeny kapcsolatok megszüntetésére. Az OpenCV cv2.erode függvényét használjuk erre a célra.
- **Dilatáció:** A dilatáció az objektum kontúrjait kibővíti. Ez a módszer segít a kis szakadások kitöltésében, valamint a háttérből kinyúló objektumok összekapcsolásában. Az OpenCV cv2.dilate függvénye biztosítja a dilatációt.
- **Nyitás (Opening):** A nyitás az erózió és dilatáció egymást követő alkalmazása, amely először eltávolítja a kisebb zajokat (erózióval), majd visszaállítja az eredeti objektum méretét (dilatációval). A nyitást gyakran használják kis fehér zaj eltávolítására.
- **Zárás (Closing):** A zárás a dilatáció és erózió egymást követő alkalmazása, amely először kibővíti az objektumokat (dilatációval), majd visszaállítja azok eredeti méretét (erózióval). Ez a módszer hasznos a kis fekete zaj eltüntetésére vagy az objektumok közötti kis hézagok kitöltésére.

#### 2.1.5. Kontúrok és bounding box-ok generálása
A `cv2.findContours` függvény az objektumok körvonalait detektálja, amelyekből bounding boxokat generálunk.

#### 2.1.6. Hisztogram kiegyenlítés
A hisztogram kiegyenlítés olyan képkorrekciós technika, amely javítja a kontrasztot azáltal, hogy az intenzitásértékeket egyenletesebben osztja el. Ez különösen hasznos lehet alacsony kontrasztú képek esetén, például gyengén megvilágított vagy árnyékos területeken készült videók feldolgozásakor.

**Alkalmazás:**
- Növeli a részletek láthatóságát a képen.
- Tisztább kontúrokat eredményezhet, amelyek javítják a detektálás pontosságát.
- Egyszerűen implementálható az OpenCV cv2.equalizeHist() függvényével.
#### 2.1.7. Éldetektálás

Az éldetektálás az objektumok határainak kiemelésére szolgál. A Canny éldetektáló algoritmus például hatékonyan képes azonosítani az éleket a képen, amelyeket később felhasználhatunk a detektálási vagy követési folyamatokban.

**Lépések:**
-	A bemeneti kép simítása zajszűrővel.
-	Élek keresése gradiensváltás alapján.
-	Az eredmény bináris kép, amely megjelöli az éleket.

#### 2.1.8. Blob detection (Objektumok felismerése)

A blob detektálás egy másik fontos technika a képfeldolgozásban. Ez a módszer a képen található foltokat vagy "blob"-okat azonosítja, amelyek jellemzően azonos intenzitásúak és alakúak.

**Felhasználás:**
-	Kis méretű objektumok, például gyalogosok vagy kerékpárosok felismerése.
-	Alapvető szegmentáció az objektumok azonosítása előtt.

#### 2.1.9. Region of Interest (ROI) kiválasztása

A képfeldolgozás során a teljes képen történő feldolgozás helyett gyakran célszerűbb egy érdeklődési területet (ROI) megadni. A projektben ezt a maszkokkal valósítottuk meg, amelyek azokra a területekre koncentrálnak, ahol a járművek várhatóan megjelennek.

**Előnyök:**
- Csökkenti a számítási költséget.
- Csak a releváns területekre koncentrál, ami javítja a pontosságot.

#### 2.1.10. Frame differencing (Képkocka különbség)

A frame differencing olyan módszer, amely egymást követő képkockák között keres különbségeket. Ez segíti a mozgás detektálását, különösen akkor, ha a háttér statikus.

**Lépések:**
1.	Az aktuális képkockát levonjuk az előzőből.
2.	A különbség bináris képpé alakítása a mozgó objektumok kiemelésére.

**Használat:**
-	Gyors és hatékony módszer mozgó objektumok detektálására statikus háttér mellett.
-	Alacsonyabb számítási igényű, mint a teljes háttérszubtrakció.


## 2.2. Objektumdetektálás

#### 2.2.1. Custom detektor
- **Módszer**: Háttérszubtrakció és kontúrkeresés.
- **Előnyök**: Egyszerű, gyors, kis számítási igény.
- **Hátrányok**: Pontatlan zsúfolt vagy változó környezetben.

#### 2.2.2. YOLO detektor
- **Módszer**: Gépi tanuláson alapuló járműdetektálás.
- **Előnyök**: Pontos, képes a különböző objektumtípusok azonosítására.
- **Hátrányok**: Nagyobb számítási igény.

### 2.3. Objektumkövetés

#### 2.3.1. Euclidean tracker
- Az objektumok középpontjai közötti Euklideszi távolságot használja.
- Egyszerű és gyors, de zsúfolt környezetekben kevésbé pontos.

#### 2.3.2. SORT (Simple Online and Realtime Tracking)
- Kalman-szűrőt és az IoU metrikát használja az objektumok pontos követésére.
- Pontosabb, mint az Euclidean Tracker, de gépigényesebb.

## 3. A megvalósítás terve és kivitelezése

### 3.1. Rendszer felépítése
**A rendszer fő komponensei:**
- **Detektorok** (Custom és YOLO): Az objektumok azonosításáért felelnek.
- **Tracker-ek** (SORT és Eukledian): Az objektumok mozgásának nyomon követése és azonosítása.
- **Felhasználói interfész** (Application): A teljes folyamat vezérlése, vizualizáció.

**Bemenetek:**
- **Videók:** media/video1.mp4, media/videoplayback.mp4.
- **Maszkok:** media/mask2.png, media/mask.png.
  
**Adatfeldolgozás:**
1.	**Videók beolvasása:** A videók képkockánként kerülnek feldolgozásra.
2.	**Detektálás:** A YOLO vagy háttérkivonásos módszer használata az objektumok azonosítására.
3.	**Követés:** Az azonosított objektumokat az IoU (SORT) vagy az Euklideszi távolság (Euclidean) segítségével követjük.
4.	**Számlálás:** Ha egy objektum középpontja áthalad egy vonalon, a számláló növekszik.

### 3.2. Rendszerarchitektúra
Az alábbi folyamatábra szemlélteti a rendszer működését:
- Videó → Detektor (Custom/YOLO) → Tracker (SORT/Euclidean) → Számlálás → Kimenet (Vizualizáció)

### 3.3. Osztályok és funkciók
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


## 4. Tesztelés
### 4.1. Tesztkörnyezet
- **Operációs rendszer**: Windows 10.
- **Hardver**: Intel i7 CPU, NVIDIA RTX 3060 GPU, 32 GB RAM.
- **Szoftver**: Python 3.9, OpenCV, NumPy, cvzone.
- **Könyvtárak**: OpenCV, NumPy, Torch, Ultralytics YOLO.

### 4.2. CUDA tesztelése
A rendszerhez tartozik egy test.py nevű fájl, amely a CUDA-kompatibilis GPU elérhetőségének tesztelésére szolgál.

A kimenet megmutatja, hogy a rendszer képes-e CUDA gyorsítást használni:
- Ha CUDA elérhető, megjeleníti a GPU nevét.
- Ha nincs elérhető GPU, "Nincs GPU" üzenet jelenik meg.

### 4.3. Tesztelési esetek
1.	**YOLO + SORT:** Pontos járműdetektálás és nyomon követés.
2.	**YOLO + Euclidean:**	Azonos működés.
3.	**Custom + SORT:** Sokkal gyorsabb, kisebb gépigényű, kevésbé pontos detektálás.
4.	**Custom + Euclidean:** Tracker módosításával nem fedezhető fel különbség.

### 4.4. Eredmények összehasonlítása


## 5. Felhasználói útmutató
1.  CUDA és cuDNN telepítése
2.	Telepítse a szükséges csomagokat: **pip install -r requirements.txt**
3.	CUDA tesztelése a **test.py** használatával.
4.	Töltse le a médiafájlokat a következő Google Drive link használatával, és helyezze őket a /media mappába: (https://drive.google.com/drive/folders/1nm1JKHnu46o-tzZtBkBxpUDp9bs3cQuw?usp=drive_link)
5.	Futtassa a programot: **Teljes.py**

**A felhasználói vezérlés:**
-	**1:** az első videó feldolgozásának elindítása.
-	**2:** A második videó feldolgozásának elindítása.
-	**ESC:** Kilépés az alkalmazásból.

## 6. Összegzés
A rendszer alkalmas valós idejű forgalomelemzésre, támogatva a modern és hagyományos technikákat. A YOLO és SORT kombináció ideális nagyobb elemszámú forgalom esetén, míg a CustomDetector és EuclideanTracker alacsony költségvetésű rendszerekben alkalmazható. 

## 7. Irodalomjegyzék
1.	Redmon, J., Farhadi, A. (2018). YOLOv3: An Incremental Improvement.
2.	Bewley, A., Ge, Z., et al. (2016). Simple Online and Realtime Tracking.
3.	OpenCV Documentation - https://opencv.org
