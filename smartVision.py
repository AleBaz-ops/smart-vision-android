import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from jnius import autoclass

# Android Speech API
PythonActivity = autoclass('org.kivy.android.PythonActivity')
Intent = autoclass('android.content.Intent')
RecognizerIntent = autoclass('android.speech.RecognizerIntent')

Window.size = (360, 640)


class SmartVisionApp(App):

    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.img = Image(size_hint=(1, 0.85))
        self.btn = Button(text="Hablar", size_hint=(1, 0.15))
        self.btn.bind(on_press=self.listen)

        self.layout.add_widget(self.img)
        self.layout.add_widget(self.btn)

        # Cámara optimizada baja resolución
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Detector ligero de rostros
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        Clock.schedule_interval(self.update, 1.0 / 15.0)  # 15 FPS
        return self.layout

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        return frame, len(faces)

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame, count = self.detect_faces(frame)

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='bgr'
        )
        texture.blit_buffer(buf, colorfmt='bgr',
                            bufferfmt='ubyte')

        self.img.texture = texture

    def listen(self, instance):
        intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        intent.putExtra(
            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
        )
        intent.putExtra(
            RecognizerIntent.EXTRA_LANGUAGE,
            "es-ES"
        )

        currentActivity = PythonActivity.mActivity
        currentActivity.startActivityForResult(intent, 0)

    def on_stop(self):
        self.capture.release()


if __name__ == "__main__":
    SmartVisionApp().run()