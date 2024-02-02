from camera import take_camera
import product as pd
import type as tp
from random import randint
import speech_recognition as sr
import winsound
import keyboard

num = randint(0,15)
winsound.PlaySound(f"C:/Users/tjral/yolov5/voice/start{num}.wav", winsound.SND_ALIAS)
while True:
    try:
        print("키보드를 누르면 음성 인식이 시작됩니다.")
        #키보드를 눌렀다면 음성 인식
        if keyboard.read_key():
            st = ''

            r = sr.Recognizer()
            mic = sr.Microphone()
            winsound.PlaySound("C:/Users/tjral/yolov5/voice/sound.wav", winsound.SND_ALIAS)
            with mic as source:
                print("인식됨")
                try:
                    audio = r.listen(source, timeout = 5, phrase_time_limit = 5)
                    text = r.recognize_google(audio, language = "ko-KR")
                    print(text)
                except sr.UnknownValueError:
                    print('음성을 인식하지 못했습니다.')
                    winsound.PlaySound("C:/Users/tjral/yolov5/voice/fail.wav", winsound.SND_ALIAS)
                    winsound.PlaySound("C:/Users/tjral/yolov5/voice/retry.wav", winsound.SND_ALIAS)
                except sr.RequestError as e:
                    print(f'에러가 발생하였습니다. 에러원인 : {e}') 
                    winsound.PlaySound("C:/Users/tjral/yolov5/voice/retry.wav", winsound.SND_ALIAS)
                except sr.WaitTimeoutError:
                    print('타임아웃')
                    winsound.PlaySound("C:/Users/tjral/yolov5/voice/fail.wav", winsound.SND_ALIAS)
                    winsound.PlaySound("C:/Users/tjral/yolov5/voice/retry.wav", winsound.SND_ALIAS)   
                else:
                    for i in ['구역', '어디', '코너', '위치']:
                        if i in text:
                            st = 'type'
                            take_camera()
                            tp.detect_type()
                    if st != 'type':
                        for i in ['상품', '뭐', '무엇', '제품']:
                            if i in text:
                                st = 'product'
                                take_camera()
                                pd.detect_product() 
                    if st != 'type' and st != 'product':
                        winsound.PlaySound("C:/Users/tjral/yolov5/voice/retry.wav", winsound.SND_ALIAS)
                        continue           
    except KeyboardInterrupt:
        print("종료")
        exit()