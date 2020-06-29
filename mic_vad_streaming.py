import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
from seleni import YouTube

logging.basicConfig(level=20)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()
                    
class Speech_Recognizer():
    
    def __init__(self, vadagr = 3, device = None, rate = 16000):
        self.vadagr = vadagr
        self.model = 'deepspeech-0.7.4-models.pbmm'
        self.device = device
        self.rate = rate
        self.yt = None
        
    def recognize(self):
        # Load DeepSpeech model
        if os.path.isdir(self.model):
            model_dir = self.model
            self.model = os.path.join(model_dir, 'output_graph.pb')
        print('Initializing model...')
        logging.info("model: %s", self.model)
        model = deepspeech.Model(self.model)
        
        # Start audio with VAD
        vad_audio = VADAudio(aggressiveness=self.vadagr,
                             device=self.device,
                             input_rate=self.rate,
                             file=None)
        print("Listening (ctrl-C to exit)...")
        frames = vad_audio.vad_collector()
    
        # Stream from microphone to DeepSpeech using VAD
        stream_context = model.createStream()
        wav_data = bytearray()
        for frame in frames:
            if frame is not None:
                logging.debug("streaming frame")
                stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            else:
                logging.debug("end utterence")
                text = stream_context.finishStream()
                print("Recognized: %s" % text)
                if text == 'youtube' or text == 'you to':
                    self.yt = YouTube()
                    self.yt.go_to()
                if self.yt != None:
                    if text == 'click trending':
                        self.yt.click_trending()
                    if text == 'close':
                        self.yt.close_youtube()
                    if text == 'previous page':
                        self.yt.previous_page()
                    if text == 'next page':
                        self.yt.next_page()
                    if text == 'create':
                        self.yt.create()
                    if text == 'next item':
                        self.yt.next_item()
                    if text == 'sign_in':
                        self.yt.sign_in()
                    if text == 'select first':
                        self.yt.select_first_account()
                    if text == 'select second':
                        self.yt.select_second_account()
                    if text == 'write password':
                        self.yt.write_password()
                    if text == 'pause':
                        self.yt.pause_or_play()    
                    if text == 'play':
                        self.yt.pause_or_play()    
                    if text == 'skip 5 seconds':
                        self.yt.skip_5_seconds() 
                    if text == 'rewind 5 seconds':
                        self.yt.rewind_5_seconds()
                    if text == 'sign out':
                        self.yt.sign_out()     
                    if text == 'toggle autoplay':
                        self.yt.toggle_autoplay()
                    if text == 'like video':
                        self.yt.like_video()
                    if text == 'dislike video':
                        self.yt.dislike_video()    
                    if text == 'search':
                        self.yt.search()
                    if text == 'theatre mode':
                        self.yt.theatre_mode()
                    if text == 'increase speed':
                        self.yt.increase_speed()
                    if text == 'decrease speed':
                        self.yt.decrease_speed()
                    if text == 'zoom out':
                        self.yt.zoom_out()   
                    if text == 'mute':
                        self.yt.mute_unmute()   
                    if text == 'unmute':
                        self.yt.mute_unmute()
                    if text == 'skip to section':
                        self.yt.skip_to_section()
                    if text == 'next video':
                        self.yt.next_video()
                    if text == 'previous video':
                        self.yt.previous_video()  
                    if text == 'full screen':
                        self.yt.full_screen() 
                    if text == 'play first video':
                        self.yt.play_first_video()
                    if text == 'play second video':
                        self.yt.play_second_video()
                    if text == 'play third video':
                        self.yt.play_third_video()    
                
                stream_context = model.createStream()
    

if __name__ == '__main__':
    recognizer  = Speech_Recognizer()
    recognizer.recognize()
