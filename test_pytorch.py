import gi
gi.require_version('Gst', '1.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst
import numpy as np
import torch
from PIL import Image
import cv2
import time

Gst.init(None)

src = Gst.ElementFactory.make('filesrc', 'source')
src.set_property('location', '/path/tp/video.mp4')  # path to video file
src.set_property('num-buffers', 9000)
decodebin = Gst.ElementFactory.make('decodebin', 'decoder')
videoconvert = Gst.ElementFactory.make('videoconvert', 'converter')
capsfilter = Gst.ElementFactory.make('capsfilter', 'capsfilter')
caps = Gst.Caps.from_string('video/x-raw,format=RGB')
capsfilter.set_property('caps', caps)

queue = Gst.ElementFactory.make('queue', 'queue')
appsink = Gst.ElementFactory.make('appsink', 's')
appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)

pipeline = Gst.Pipeline.new('new-pipeline')

pipeline.add(src)
pipeline.add(decodebin)
pipeline.add(capsfilter)
pipeline.add(videoconvert)
pipeline.add(queue)
pipeline.add(appsink)

src.link(decodebin)
decodebin.link(videoconvert)
videoconvert.link(capsfilter)
capsfilter.link(queue)
queue.link(appsink)

def on_pad_added(element, pad):
    sinkpad = videoconvert.get_static_pad('sink')
    pad.link(sinkpad)
    
decodebin.connect('pad-added', on_pad_added)

device = torch.device('cpu')
print(device)
detector = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt').eval().to(device)

fps = 0
prev_time = time.time()

def on_new_sample(appsink):
    global fps, prev_time
    sample = appsink.emit('pull-sample')
    buf = sample.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')
    caps = sample.get_caps()
    image_tensor = buffer_to_image_tensor(buf, caps)
    
    with torch.no_grad():
        detections = detector(image_tensor)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        print("FPS: {0:.2f}".format(fps))
        prev_time = current_time
        print(detections)
        objects = (detections.xyxy[0]).tolist()
        img_np = np.array(image_tensor)
        for i in range(len(objects)):
            x_min, y_min, x_max, y_max = int(objects[i][0]), int(objects[i][1]), int(objects[i][2]), int(objects[i][3])
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
            label = detector.names[int(objects[i][5])]
            score = objects[i][4]
            cv2.putText(
                img_np, f'{label}: {score:.2f}', (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )
        # Save the output image
        cv2.imshow('output', img_np)
        cv2.waitKey(1)
    return Gst.FlowReturn.OK

appsink.connect('new-sample', on_new_sample)


def buffer_to_image_tensor(buf, caps):
    caps_structure = caps.get_structure(0)
    height, width = caps_structure.get_value('height'), caps_structure.get_value('width')
    channels = 3
    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if is_mapped:
        try:
            image_array = np.frombuffer(map_info.data, dtype=np.uint8).reshape((width, height, channels)).copy()
            image_array.resize((height, width, channels))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image_array, (1920, 1080))
            return Image.fromarray(image_array) # RGBA -> RGB
        finally:
            buf.unmap(map_info)

# pipeline.get_by_name('s').get_static_pad('sink').add_probe(
#     Gst.PadProbeType.BUFFER,
#     on_frame_probe
# )

pipeline.set_state(Gst.State.PLAYING)

while True:
    msg = pipeline.get_bus().timed_pop_filtered(
        Gst.SECOND,
        Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    if msg:
        text = msg.get_structure().to_string() if msg.get_structure() else ''
        msg_type = Gst.message_type_get_name(msg.type)
        print(f'{msg.src.name}: [{msg_type}] {text}')
        break

pipeline.set_state(Gst.State.NULL)