# Тема ВКР: Разработка программного обеспечения сегментации и распознавания объектов игровой ситуации в волейболе
# Автор: Пасынков М.И.
# Направление: 09.03.04 Программная инженерия
# Группа: 9413
# Руководитель ВКР: к.т.н., доц. каф. ВПМ Цуканова Н.И.
# Средства разработки:
#    ОС - Windows 10
#    Язык программирования - Python 3.9
#    Среды разработки: PyCharm Community Edition 2021.2.2, Google Collaboratory
# Назначение: интерфейс программы и функция отслеживания перемещений игроков
# Дата разработки: 19.04.2023

# Импорт программных модулей
import random
import threading
import time
import cv2
from ultralytics import YOLO
from tracker import Tracker
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, \
    QSlider, QStyle, QSizePolicy, QFileDialog, QDialog, QDialogButtonBox, QCheckBox
import sys
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl, QPointF
import pyqtgraph as pg
import shutil

# Отслеживает перемещения игроков на видеозаписи
# Входные данные:
#     videoPath - путь к исходному видео
#     label - элемент пользовательского интерфейса,
#         на который выводится прогресс обработки
#     points - крайние точки волейбольной площадки
# Выходные данные:
#     videoOutPath - путь к обработанному видефайлу
def track_video(videoPath, label, points=[]):

    # Расчет координат площадки
    if points != []:
        points.sort(key=lambda x: x[0])
        # Первая координата верхнего левого угла
        areax1 = (points[0][0] + points[1][0]) // 2
        # Первая координата нижнего правого угла
        areax2 = (points[2][0] + points[3][0]) // 2
        points.sort(key=lambda x: x[1])
        # Вторая координата верхнего левого угла
        areay1 = (points[0][1] + points[1][1]) // 2
        # Вторая координата нижнего правого угла
        areay2 = (points[2][1] + points[3][1]) // 2
        # Коэффициенты пустого пространства за границами площадки
        # справа и слева
        sidesX = 0.2
        # сверху и снизу
        sidesY = 0.3

    # Путь для сохранения обработанного видеофайла
    videoOutPath = 'D:/TrackingProject/out.mp4'

    # Инициализация входного видеопотока
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()

    # Общее количество кадров в видеофайле
    framesNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Номер текущего кадра
    frameNum = 1

    # Инициализация выходного видеопотока
    cap_out = cv2.VideoWriter(videoOutPath, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                              (frame.shape[1], frame.shape[0]))

    # Расчет координат площадки на выходно видеозаписи
    # в случае перевода в схематический формат
    if points != []:
        # Количество пикселей по координате Х,
        # которые соответствуют длине площадки
        delX = frame.shape[1] / (1 + sidesX * 2)
        # Первая координата левого верхнего угла площадки
        newAreaX1 = int(delX * sidesX)
        # Первая координата правого нижнего угла площадки
        newAreaX2 = int(newAreaX1 + delX)
        # Количество пикселей по координате Y,
        # которые соответствуют ширине площадки
        delY = frame.shape[0] / (1 + sidesY * 2)
        # Вторая координата левого верхнего угла площадки
        newAreaY1 = int(delY * sidesY)
        # Вторая координата правого нижнего угла площадки
        newAreaY2 = int(newAreaY1 + delY)
        # Отношение для переноса координат Х на новый кадр
        ratioX = (newAreaX2 - newAreaX1) / (areax2 - areax1)
        # Отношение для переноса координат Y на новый кадр
        ratioY = (newAreaY2 - newAreaY1) / (areay2 - areay1)

    # Модель нейронной сети YOLO
    model = YOLO("D:/TrackingProject/best.pt")

    # Объект для отслеживания
    tracker = Tracker()

    # Список цветов для отображения ограничивающих прямоугольников или окружностей
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(13)]

    # Пороговый уровень вероятности при обнаружении игроков
    detection_threshold = 0.4
    while ret:

        # Обработка кадра нейронной сетью
        results = model(frame)
        # Кадр для записи в выходной видеопоток
        newFrame = frame
        # Заполнение кадра белым цветом
        if points != []:
            newFrame.fill(255)

        for result in results:
            # Результаты детектирования
            detections = []
            for r in result.boxes.data.tolist():
                # Получение координат, вероятности и номера класса
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                # Добавление результата детектирования,
                # если превышен порог вероятности
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

            # Обновление идентификаторов объектов отслеживания
            tracker.update(frame, detections)

            # Отрисовка объектов
            for track in tracker.tracks:
                # Ограничивающий прямоугольник
                bbox = track.bbox
                # Координаты левого верхнего и правого нижнего угла прямоугольника
                x1, y1, x2, y2 = bbox
                # Идентификатор объекта
                track_id = track.track_id

                # Если не заданы точки площадки, то отрисовка прямоугольников
                # иначе - отрисовка окружностей
                if points == []:
                    cv2.rectangle(newFrame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                    cv2.putText(newFrame, str(track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (colors[track_id % len(colors)]), 2)
                else:
                    # Координата Х центра окружности
                    circleX = int(newAreaX1 + (int(((int(x1) + int(x2)) / 2 - areax1) * ratioX)))
                    # Координата Y центра окружности
                    circleY = int(newAreaY1 + ((int(y2) - areay1) * ratioY))
                    cv2.circle(newFrame, (circleX, circleY), 30, (colors[track_id % len(colors)]), 3)
                    cv2.putText(newFrame, str(track_id), (circleX - 15, circleY - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (colors[track_id % len(colors)]), 2)
            # Если заданы координаты площадки, отрисовка площадки
            if points != []:
                cv2.rectangle(newFrame, (newAreaX1, newAreaY1), (newAreaX2, newAreaY2), (0, 0, 0), 3)

        # Запись кадра в выходной поток
        cap_out.write(newFrame)
        # Чтение нового кадра из входного потока
        ret, frame = cap.read()
        # Вывод прогресса
        label.setText('Прогресс обработки: ' + str(frameNum * 100 // framesNum) + '%')
        # Увеличение счетчика текущего кадра
        frameNum += 1

    # Освобождение потоков
    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()

    # Возврат пути к обработанному файлу
    return videoOutPath




# Область изображения
class ImagePlot(pg.GraphicsLayoutWidget):

    # Инициализация объекта
    def __init__(self):
        super(ImagePlot, self).__init__()
        self.p1 = pg.PlotItem()
        self.addItem(self.p1)
        # Инвертирование изображение по координате Y
        # для корректного отображение
        self.p1.vb.invertY(True)

        # Инициализация объекта ScatterPlotItem
        # для отрисовки точек на изображении
        self.scatterItem = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0),
            hoverable=True,
            hoverBrush=pg.mkBrush(0, 255, 255)
        )
        # Вывод элемента на передний план
        self.scatterItem.setZValue(2)
        # Список точек, выбранных пользователем
        self.points = []

        # Добавление элемента scatterItem к области изображения
        self.p1.addItem(self.scatterItem)

    # Установка изображения
    # Входные данные:
    #     img - изображение
    def setImage(self, img):
        # Очистка области изображения
        self.p1.clear()
        # Добавление элемента scatterItem
        self.p1.addItem(self.scatterItem)
        # Инициализация изображения
        self.imageItem = pg.ImageItem(img)
        # Настройка осей координат
        self.imageItem.setOpts(axisOrder='row-major')
        # Подключение изображения к области
        self.p1.addItem(self.imageItem)

    # Обработчик события нажатия ЛКМ
    # Входные данные:
    #     event - данные о событии
    def mousePressEvent(self, event):
        # Инициализация точки по координатам нажатия ЛКМ
        point = self.p1.vb.mapSceneToView(QPointF(event.pos()))  # get the point clicked
        # Получение координат нажатия ЛКМ
        x, y = int(point.x()), int(point.y())

        # Сброс уже добавленных точек, если их 4
        if len(self.points) == 4:
            self.points.clear()
            self.scatterItem.clear()

        # Добавление точки в список точек
        self.points.append([x, y])
        # Добавление точки к элементу scatterItem
        self.scatterItem.addPoints(pos=self.points)
        super().mousePressEvent(event)

# Диалог для выбора крайних точек площадки
class AreaPointsDialog(QDialog):
    # Инициализация
    # Входные данные:
    #     videoPath - путь к входному видеофайлу
    #     parent - родительский элемент
    def __init__(self, videoPath, parent=None):
        super().__init__(parent)

        # Установка заголовка окна
        self.setWindowTitle("Выбор крайних точек площадки")

        # Инициализация кнопок закрытия диалога
        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        # Инициализация контейнера кнопок
        self.buttonBox = QDialogButtonBox(QBtn)
        # Привязка обработчика при нажатии кнопки Ok
        self.buttonBox.accepted.connect(self.pressOkBtn)
        # Привязка обработчика при нажатии кнопки Cancel
        self.buttonBox.rejected.connect(self.reject)

        # Инициализация вертикального контейнера
        self.layout = QVBoxLayout()
        # Инициализация области изображения
        self.imagePlot = ImagePlot()
        # Добавление области изображения в контейнер
        self.layout.addWidget(self.imagePlot)

        # Получение первого кадра видеозаписи
        cap = cv2.VideoCapture(videoPath)
        res, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        # Установка кадра в область изображенияя
        self.imagePlot.setImage(frame)

        # Создание поясняющей надписи на форме
        self.label = QLabel()
        self.label.setText('Выберите 4 крайние точки волейбольной площадки.')
        self.layout.addWidget(self.label)

        # Добавление контейнера кнопок
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    # Обработчик события нажатия кнопки Ok
    def pressOkBtn(self):
        # Если выбраны 4 точки, форма закрывается,
        # иначе - выводится сообщение
        if len(self.imagePlot.points) == 4:
            self.accept()
        else:
            self.label.setText('Необходимо выбрать ровно 4 точки!')

# Главное окно
class Window(QWidget):

    # Инициализация
    def __init__(self):
        super().__init__()

        # Установка заголовка окна
        self.setWindowTitle("Волейбол-трекер")
        # Установка размеров окна
        self.setGeometry(350, 100, 700, 500)

        # Инициализация цветовой палитры
        p = self.palette()
        self.setPalette(p)

        # Инициализация полей
        self.fileName = ''
        self.areaPoints = []

        # Инициализация элементов окна
        self.init_ui()
        # Вывод окна на экран
        self.show()

    # Инициализация элементов окна
    def init_ui(self):

        # Создание объекта Media Player
        self.mediaPlayer = QMediaPlayer()
        # Создание объекта Video Widget
        videoWidget = QVideoWidget()

        # Создание кнопки открытия видео
        self.openBtn = QPushButton('Открыть видео')
        # Привязка обработчика события нажатия кнопки
        self.openBtn.clicked.connect(self.openFile)

        # Создание кнопки обработки видеозаписи нейронной сетью
        self.execBtn = QPushButton('Обработать')
        # Кнопка недоступна, пока не открыто видео
        self.execBtn.setEnabled(False)
        # Привязка обработчика события нажатия кнопки
        self.execBtn.clicked.connect(self.execution)

        # Создание флажка перевода в схематический формат
        self.typeChBx = QCheckBox('Перевести в графический формат')

        # Создание кнопки сохранения
        self.saveBtn = QPushButton('Сохранить видео')
        # Кнопка недоступна, пока не открыто видео
        self.saveBtn.setEnabled(False)
        # Привязка обработчика события нажатия кнопки
        self.saveBtn.clicked.connect(self.saveFile)

        # Создание кнопки воспроизведения/паузы
        self.playBtn = QPushButton()
        # Кнопка недоступна, пока не открыто видео
        self.playBtn.setEnabled(False)
        # Установка иконки
        self.playBtn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        # Привязка обработчика события нажатия кнопки
        self.playBtn.clicked.connect(self.playVideo)

        # Создание ползунка
        self.slider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.slider.setRange(0,0)
        # Привязка обработчика события передвижения ползунка
        self.slider.sliderMoved.connect(self.setPosition)

        # Создание надписи для вывода прогресса обработки
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # Создание контейнеров и расположение элементов в них
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0,0,0,0)
        hboxLayout.addWidget(self.openBtn)
        hboxLayout.addWidget(self.saveBtn)
        hboxLayout.addWidget(self.execBtn)
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)

        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(videoWidget)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.typeChBx)
        vboxLayout.addWidget(self.label)

        # Привязка контейнера к форме
        self.setLayout(vboxLayout)
        # Привязка элемента Video Widget к Media Player
        self.mediaPlayer.setVideoOutput(videoWidget)

        # Привязка обработчиков событий объекта Media Player
        self.mediaPlayer.playbackStateChanged.connect(self.mediastateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionСhanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

    # Обработчик события нажатия кнопки открытия файла
    def openFile(self):
        # Получение пути к файлу с помощью диалога
        filename, _ = QFileDialog.getOpenFileName(self, "Открыть видео", filter="Видеофайлы (*.mp4)")

        # Если файл выбран, загрузка файла в Media Player
        # и открытие доступа к кнопкам
        if filename != '':
            self.mediaPlayer.setSource(QUrl.fromLocalFile(filename))
            self.playBtn.setEnabled(True)
            self.execBtn.setEnabled(True)
            self.saveBtn.setEnabled(True)

        # Сохранение пути к файлу в поле
        self.fileName = filename

    # Обработчик события нажатия кнопки сохранения в файл
    def saveFile(self):
        # Получение директории для сохранения файла с помощью диалога
        filename, _ = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")

        # Копирование выходного файла в выбранную категорию
        if filename != '':
            shutil.copy(self.fileName, filename)

    # Обработчик события нажатия кнопки обработки видео
    def execution(self):
        # Очистка крайних точек площадки (если были добавлены ранее)
        self.areaPoints.clear()
        if self.typeChBx.isChecked():
            # Получение новых точек
            dlg = AreaPointsDialog(self.fileName)
            if dlg.exec():
                self.areaPoints = dlg.imagePlot.points

        # Создание и запуск потока обработки видеозаписи
        t1 = threading.Thread(target=self.execVideo, args=(), daemon=True)
        t1.start()

    # Обработчик, запускаемый потоком обработки видеозаписи
    def execVideo(self):
        # Запретить доступ к кнопкам на время обработки
        self.saveBtn.setEnabled(False)
        self.execBtn.setEnabled(False)
        self.openBtn.setEnabled(False)
        self.playBtn.setEnabled(False)

        # Получение пути к выходному файлу после обработки
        outFileName = track_video(self.fileName, self.label, self.areaPoints)

        # Возврат доступа к кнопкам после обработки
        self.saveBtn.setEnabled(True)
        self.execBtn.setEnabled(True)
        self.openBtn.setEnabled(True)
        self.playBtn.setEnabled(True)

        # Вывод надписи о завершении обработки
        self.label.setText('Обработка завершена.')

        # Подключение полученного видео к Media Player
        if outFileName != '':
            self.mediaPlayer.setSource(QUrl.fromLocalFile(outFileName))
            self.playBtn.setEnabled(True)
            self.execBtn.setEnabled(False)

    # Обработки события нажатия кнопки запуска/остановки видео
    def playVideo(self):
        # Если видео воспроизводится - остановить,
        # иначе - запустить
        if self.mediaPlayer.isPlaying():
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    # Обработчик изменения состояния объекта Media Player
    def mediastateChanged(self, state):
        # Изменение иконки кнопки воспроизведения/остановки
        if self.mediaPlayer.isPlaying():
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            )

        else:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )

    # Обработчик события изменения позиции воспроизведения видео
    # Входные данные:
    #     position - текущая позиция видеозаписи
    def positionСhanged(self, position):
        # Изменение положения ползунка
        self.slider.setValue(position)

    # Обработчик события изменения продолжительности видео
    # Входные данные:
    #     duration - продолжительность видео
    def durationChanged(self, duration):
        self.slider.setRange(0, duration)

    # Обработчик события изменения положения ползунка
    # Входные данные:
    #     position - позиция ползунка
    def setPosition(self, position):
        # Изменить позицию видеозаписи
        self.mediaPlayer.setPosition(position)

    # Обработчик события возникновения ошибки
    def handleErrors(self):
        # Запрет доступа к кнопке воспроизведения
        self.playBtn.setEnabled(False)
        # Вывод сообщения об ошибке
        self.label.setText("Error: " + self.mediaPlayer.errorString())

# Инициализация приложения
app = QApplication(sys.argv)
# Инициализация главного окна
window = Window()
# Запуск приложения
sys.exit(app.exec())