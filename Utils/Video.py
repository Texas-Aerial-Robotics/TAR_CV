import cv2
import numpy as np
import gi

# Require GStreamer version 1.0
gi.require_version('Gst', '1.0')
from gi.repository import Gst


class Video:
    """
    A class to capture a GStreamer video stream, convert it to an OpenCV format,
    and make it available for processing. This is particularly useful for
    receiving video streams from sources like the Gazebo simulator.
    """
    def __init__(self, port=5600):
        """
        Initializes the GStreamer pipeline and starts video capture.
        
        Args:
            port (int): The UDP port to listen on for the video stream.
        """
        # Initialize the GStreamer library
        Gst.init(None)
        
        self.port = port
        self._frame = None  # This will store the latest captured frame

        # --- GStreamer Pipeline Configuration ---
        # Define the different stages of the GStreamer pipeline as separate strings.
        # This makes the pipeline easier to read and modify.

        # 1. Video Source: Listen for a UDP stream on the specified port.
        self.video_source = f'udpsrc port={self.port}'
        
        # 2. Video Codec and Depayloading:
        #    - 'application/x-rtp': Specifies the incoming data is in RTP format.
        #    - 'rtph264depay': Extracts the H.264 video data from the RTP packets.
        #    - 'avdec_h264': Decodes the H.264 compressed video into raw video frames.
        self.video_codec = '! application/x-rtp ! rtph264depay ! avdec_h264'
        
        # 3. Video Decoding and Color Space Conversion:
        #    - 'videoconvert': Converts the video frames to a different format.
        #    - 'video/x-raw,format=BGR': Specifies the output format should be raw video
        #      with BGR color space, which is the standard format used by OpenCV.
        self.video_decode = '! videoconvert ! video/x-raw,format=BGR'
        
        # 4. Video Sink: The endpoint of the pipeline.
        #    - 'appsink': Makes the video frames available to the application (our Python script)
        #      instead of rendering them to a screen.
        #    - 'emit-signals=true': Tells the appsink to emit a 'new-sample' signal
        #      whenever a new frame is ready.
        #    - 'sync=false': Prevents the sink from synchronizing on the clock, processing
        #      frames as fast as they arrive.
        self.video_sink_conf = '! appsink name=appsink0 emit-signals=true sync=false'

        self.video_pipe = None  # Will hold the GStreamer pipeline object
        self.video_sink = None  # Will hold the appsink element
        
        # Start the pipeline immediately upon initialization
        self.run()

    def start_gst(self, config=None):
        """
        Builds and starts a GStreamer pipeline from a configuration list.
        
        Args:
            config (list, optional): A list of strings that define the pipeline.
                                     Defaults to a test source if not provided.
        """
        if not config:
            # Default pipeline for testing: generates a color bar pattern.
            config = [
                'videotestsrc ! decodebin',
                '! videoconvert ! video/x-raw,format=BGR',
                '! appsink'
            ]
        
        # Join the configuration list into a single command string for GStreamer
        command = ' '.join(config)
        
        # Parse the command string to create the GStreamer pipeline object
        self.video_pipe = Gst.parse_launch(command)
        
        # Start the pipeline
        self.video_pipe.set_state(Gst.State.PLAYING)
        
        # Get the appsink element by its name so we can connect to its signals
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """
        A static method to convert a GStreamer sample to an OpenCV BGR image (NumPy array).
        
        Args:
            sample: A Gst.Sample object from the appsink.
            
        Returns:
            np.ndarray: The video frame as a NumPy array.
        """
        # Get the raw data buffer from the GStreamer sample
        buf = sample.get_buffer()
        
        # Get the video's properties (width, height) from the sample's capabilities
        caps = sample.get_caps()
        
        # Create a NumPy array from the raw buffer
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3  # 3 color channels (BGR)
            ),
            buffer=buf.extract_dup(0, buf.get_size()), # Copy buffer data
            dtype=np.uint8
        )
        return array

    def frame(self):
        """
        Public method to get the most recent frame.
        
        Returns:
            np.ndarray or None: The latest frame as a NumPy array, or None if no frame is available.
        """
        return self._frame

    def frame_available(self):
        """
        Public method to check if a frame has been received.
        
        Returns:
            bool: True if a frame is available, False otherwise.
        """
        return self._frame is not None

    def run(self):
        """
        Constructs and starts the main video capture pipeline and connects the callback.
        """
        # Assemble the full pipeline configuration from the components
        pipeline_config = [
            self.video_source,
            self.video_codec,
            self.video_decode,
            self.video_sink_conf
        ]
        self.start_gst(pipeline_config)
        
        # Connect the 'new-sample' signal from the appsink to our 'callback' method.
        # This means self.callback will be called automatically every time a new frame arrives.
        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        """
s        The callback function that is executed automatically on a new frame.
        
        Args:
            sink: The Gst.AppSink element that emitted the signal.
            
        Returns:
            Gst.FlowReturn.OK: Required by GStreamer to indicate successful processing.
        """
        # Pull the Gst.Sample object from the sink
        sample = sink.emit('pull-sample')
        
        # Convert the GStreamer sample to an OpenCV-compatible NumPy array
        new_frame = self.gst_to_opencv(sample)
        
        # Update the class's internal frame storage
        self._frame = new_frame
        
        # Indicate to GStreamer that the frame was processed correctly
        return Gst.FlowReturn.OK
