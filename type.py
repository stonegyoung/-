import cv2 as cv
import winsound
import argparse
from pathlib import Path


from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
import winsound
import torch

def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='real.pt', help='model path or triton URL')
        parser.add_argument('--source', type=str, default='img_captured.png', help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default= './data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold') #좀 느림
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt



########################구역 인식#################################################    
def play(list1):
    list1.append('is')
    cnt = 0
    while(cnt<len(list1)):
        winsound.PlaySound(f"C:/Users/tjral/yolov5/voice/{list1[cnt]}.wav", winsound.SND_ALIAS)
        cnt += 1
                
def classification(arr):
    product = {"snack": 0, "ramen": 0, "beverage" :0}
    p_type = {'Cocacola': "beverage", 'JinRamen': "ramen", 'Jjapagheti': "ramen",
    'Powerade': "beverage", 'SinRamen': "ramen",
    'Choco Heim': "snack", 'potatochip': "snack", 
    'saeukkang hot': "snack", 'sprite': "beverage"}
    
    sn, rm, bv = 0, 0, 0    #x1 인덱스 합
    
    if len(arr) == 0:
        print("아무것도 인식되지 않았습니다\n")
        winsound.PlaySound("C:/Users/tjral/yolov5/voice/nothing.wav", winsound.SND_ALIAS)
    elif len(arr) == 1:
        print(f"{p_type[next(iter(arr))]} 구역 입니다\n")
        i = p_type[next(iter(arr))]
        play([i])
    else:
        for i in arr.keys():
            if p_type[i] == "snack":
                product["snack"] += 1
                sn += sum(arr[i])
            elif p_type[i] == "beverage":
                product["beverage"] += 1
                bv += sum(arr[i])
            elif p_type[i] == "ramen":
                product["ramen"] += 1
                rm += sum(arr[i])
            else:
                pass
        many_type = [k for k,v in product.items() if max(product.values()) == v] #리스트
        #print(many_type)
        if len(many_type) == 1:
            print(f"{many_type} 구역 입니다\n")
            play(many_type)
        else:
            #ramen, beverage, snack 오름차순 정렬
            ll = {'ramen': rm, 'beverage': bv, 'snack' : sn}
            ll = dict(sorted(ll.items(), key=lambda x:x[1]))
            str = []
            for i in ll.keys():
                str.append(i)
            if len(many_type) == 2:
                #왼쪽 오른쪽 나누기
                a = ['left', str[1], 'right', str[2]]
                play(a)
                # a.append(str[0])
                print(f"왼쪽 {str[1]} 구역, 오른쪽 {str[2]} 구역입니다")
            elif len(many_type) == 3:
                #왼쪽 중간 오른쪽 나누기
                a = ['left', str[0], 'center', str[1], 'right', str[2]]
                play(a)
                print(f"왼쪽 {str[0]} 구역, 가운데 {str[1]} 구역, 오른쪽 {str[2]} 구역입니다")


def detect_type():
    
    @smart_inference_mode()
    def run(
            weights='real.pt',  # model path or triton URL
            source='img_captured.png',  # file/dir/URL/glob/screen/0(webcam)
            data='./data/coco128.yaml', # dataset.yaml path
            imgsz=(480, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_csv=False,  # save results in CSV format
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project= './runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        source = str(source)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # runs/detect/exp38
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, dt = 0, (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            arr ={}
            for det in (pred):  # per image
                seen += 1
                p, im0 = path, im0s.copy()

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()


                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        #global arr
                        arr[f"{names[int(c)]}"]=[]

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f'{names[c]}'
                        
                        
                        x1 = int(xyxy[0].item()) # bounding box 좌표 x1
            
                        arr[label].append(x1)

                        
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
                # Save results (image with detections) 결과 저장
                cv2.imwrite(save_path, im0)
            #print(arr)
            classification(arr)
            
        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
        # if update:
        #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

          
    opt = parse_opt()
    run(**vars(opt))