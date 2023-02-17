import os, sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import numpy as np
import torch
from PIL import Image
import cv2

frame_format, pixel_bytes = 'RGBA', 4

Gst.init(None)
# pipeline = Gst.parse_launch(f'''
#     filesrc location=./video/PEDESTRIAN/KM77000.VDS.CAM.17_10.2.163.76_20221227T150312_PEDESTRIAN_1_ACCEPTED.avi num-buffers=200 !
#     decodebin !
#     videoconvert !
#     autovideosink name = s
# ''')

pipeline = Gst.Pipeline.new('new-pipeline')

src = Gst.ElementFactory.make('filesrc', 'source')
src.set_property('location', './video/test.mp4')
src.set_property('num-buffers', 200)
decodebin = Gst.ElementFactory.make('decodebin', 'decoder')
videoconvert = Gst.ElementFactory.make('videoconvert', 'converter')
autovideosink = Gst.ElementFactory.make('autovideosink', 's')

s = pipeline.get_by_name('s')

pipeline.add(src)
pipeline.add(decodebin)
pipeline.add(videoconvert)
pipeline.add(autovideosink)

def on_pad_added(element, pad):
    sinkpad = videoconvert.get_static_pad('sink')
    pad.link(sinkpad)
    sinkpad = autovideosink.get_static_pad('sink')
    videoconvert.link(autovideosink)
    
src.link(decodebin)
decodebin.connect('pad-added', on_pad_added)

detector = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt').eval().to(torch.device('cpu'))

def on_frame_probe(pad, info):
    # sample = s.emit('pull-sample')
    buf = info.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')
    image_tensor = buffer_to_image_tensor(buf, pad.get_current_caps())
    # print(image_tensor.size)
    with torch.no_grad():
        detections = detector(image_tensor)
        objects = (detections.xyxy[0]).tolist()
        img_np = np.array(image_tensor)
        for i in range(len(objects)):
            x_min, y_min, x_max, y_max = int(objects[i][0]), int(objects[i][1]), int(objects[i][2]), int(objects[i][3])
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
        # print(img_np.shape)
        
    new_data = img_np.tobytes()
    new_buf = Gst.Buffer.new_wrapped(new_data)
    sample_new = Gst.Sample.new(new_buf, pad.get_current_caps())
    s.emit('push-sample', sample_new)
        
    return Gst.PadProbeReturn.OK

def buffer_to_image_tensor(buf, caps):
    caps_structure = caps.get_structure(0)
    height, width = caps_structure.get_value('height'), caps_structure.get_value('width')
    # print(height, width)
    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if is_mapped:
        try:
            # image_array = np.ndarray(
            #     (height, width, pixel_bytes),
            #     dtype=np.uint8,
            #     buffer=map_info.data
            # ).copy() # extend array lifetime beyond subsequent unmap
            image_array=np.frombuffer(map_info.data ,dtype=np.uint8).reshape(540, width,3)
            # print(image_array.shape)
            return Image.fromarray(image_array[:,:,:3]) # RGBA -> RGB
        finally:
            buf.unmap(map_info)

pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)

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