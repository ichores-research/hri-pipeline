import rospy
import pyaudio
import threading
import time
from audio_common_msgs.msg import AudioData, AudioDataStamped
from std_msgs.msg import Header

def get_pyaudio_format(format_str):
  format_mapping = {
    'S16_LE': pyaudio.paInt16,
    'S32_LE': pyaudio.paInt32,
    'FLOAT': pyaudio.paFloat32,
    'U8': pyaudio.paUInt8,
    'S8': pyaudio.paInt8,
    'ALAW': pyaudio.paALaw,
    'ULAW': pyaudio.paULaw,
  }
  
  return format_mapping.get(format_str, None)

class AudioPublisher:
  def __init__(self):
    rospy.init_node('mic_publisher_node', anonymous=True)
    
    # ROS Parameters
    self.ros_params = rospy.get_param("mic_analyzer")
    self.device_id = self.ros_params.get("device_id", "(hw:0,0)")
    self.sample_rate = self.ros_params.get("input_sample_rate", 48000)
    self.chunk_size = self.ros_params.get("chunk_size", 1024)
    self.channels = self.ros_params.get("num_channels", 1)
    self.format = get_pyaudio_format(self.ros_params.get("format", "S16_LE"))
    self.audio_topic = self.ros_params.get("audio_topic", "audio/raw_dji")
    
    # State variables
    self.running = True
    self.stream = None
    self.audio = pyaudio.PyAudio()
    
    # ROS Publisher
    self.audio_pub = rospy.Publisher(self.audio_topic, AudioData, queue_size=50)
    
    # Statistics for monitoring
    self.packets_sent = 0
    self.last_stats_time = time.time()
    
    # Initialize audio stream
    if not self.init_audio_stream():
      rospy.logerr("Failed to initialize audio stream. Exiting.")
      return
    
    rospy.loginfo("Audio Publisher initialized")
    rospy.loginfo("Device: {}".format(self.device_name))
    rospy.loginfo("Sample Rate: {}Hz, Channels: {}"
                  .format(self.sample_rate,self.channels))
    rospy.loginfo("Publishing to: {}".format(self.audio_topic))
    
  def init_audio_stream(self):
    """Initialize PyAudio stream for given device"""
    try:
      # Find PAL_ANDREA device
      device_index = None
      rospy.loginfo("Scanning for audio devices...")
      
      for i in range(self.audio.get_device_count()):
        info = self.audio.get_device_info_by_index(i)
        rospy.loginfo("Device {}: {} (inputs: {})"
                      .format(i,info['name'],info['maxInputChannels']))
        
        if self.device_name in info['name'] and info['maxInputChannels'] > 0:
          device_index = i
          rospy.loginfo("Selected audio device: {} (index: {})"
                        .format(info['name'],i))
          break
      
      if device_index is None:
        rospy.logerr("Could not find audio device containing '{}'"
                     .format(self.device_name))
        rospy.logerr("Available input devices:")
        for i in range(self.audio.get_device_count()):
          info = self.audio.get_device_info_by_index(i)
          if info['maxInputChannels'] > 0:
            rospy.logerr("  - {}".format(info['name']))
        return False
      
      # Test if the device supports our desired format
      try:
        self.audio.is_format_supported(
          rate=self.sample_rate,
          input_device=device_index,
          input_channels=self.channels,
          input_format=self.format
        )
      except ValueError as e:
        rospy.logwarn("Format not supported, trying alternatives: {}".format(e))
        # Try different sample rates
        for rate in [44100, 48000, 22050, 8000]:
          try:
            self.audio.is_format_supported(
              rate=rate,
              input_device=device_index,
              input_channels=self.channels,
              input_format=self.format
            )
            rospy.loginfo("Using alternative sample rate: {}".format(rate))
            self.sample_rate = rate
            break
          except:
            continue
            
      self.stream = self.audio.open(
        format=self.format,
        channels=self.channels,
        rate=self.sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=self.chunk_size,
        stream_callback=None  # We'll use blocking read
      )
      
      rospy.loginfo("Audio stream initialized successfully")
      return True
      
    except Exception as e:
      rospy.logerr("Failed to initialize audio stream: {}".format(e))
      return False
  
  def audio_publisher_loop(self):
    """Main audio capture and publishing loop"""
    rospy.loginfo("Starting audio capture thread")
    
    while not rospy.is_shutdown() and self.running:
      try:
        # Read audio data (blocking)
        audio_data = self.stream.read(
          self.chunk_size, 
          exception_on_overflow=False
        )
        # Create ROS message
        audio_msg = AudioData()
        audio_msg.data = audio_data
        self.audio_pub.publish(audio_msg)
        self.packets_sent += 1
        
        # Log statistics every 10 seconds
        current_time = time.time()
        if current_time - self.last_stats_time >= 10.0:
          rate = self.packets_sent / (current_time - self.last_stats_time)
          rospy.loginfo("Audio streaming: {:.1f} packets/sec, {} total packets"
                        .format(rate,self.packets_sent))
          self.packets_sent = 0
          self.last_stats_time = current_time
          
      except Exception as e:
        rospy.logerr("Audio capture error: {}".format(e))
        # Try to recover
        time.sleep(0.1)
        if not self.stream.is_active():
          rospy.logwarn("Stream inactive, attempting to restart...")
          self.cleanup_stream()
          if not self.init_audio_stream():
            rospy.logerr("Failed to restart audio stream")
            break
  
  def cleanup_stream(self):
    """Clean up audio stream resources"""
    if self.stream:
      try:
        if self.stream.is_active():
          self.stream.stop_stream()
        self.stream.close()
        self.stream = None
      except:
        pass
  
  def run(self):
    """Main execution function"""
    if self.stream is None:
      rospy.logerr("Audio stream not initialized. Exiting.")
      return
    
    audio_thread = threading.Thread(target=self.audio_publisher_loop)
    audio_thread.daemon = True
    audio_thread.start()
    
    try:
      rospy.spin()
    except KeyboardInterrupt:
      rospy.loginfo("Shutdown requested")
    finally:
      self.cleanup()
  
  def cleanup(self):
    """Clean up all resources"""
    rospy.loginfo("Shutting down Audio Publisher")
    self.running = False
    self.cleanup_stream()
    self.audio.terminate()
    rospy.loginfo("Audio Publisher stopped")

if __name__ == '__main__':
  try:
    publisher = AudioPublisher()
    publisher.run()
  except rospy.ROSInterruptException:
    pass