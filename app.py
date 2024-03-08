# Importing gradio for demo application and Transformers to use pipeline
import gradio as gr
from transformers import pipeline

# Using the "whisper-large-v3" fine-tuned model for Automatic Speech Recognition ASR tasks
asr = pipeline(task = "automatic-speech-recognition",
               model = "openai/whisper-large-v3")


# Set up a Gradio application with the Blocks class, which can be used to define and configure input and output blocks for the application's interface
import gradio as gr
demo = gr.Blocks()

#  Perform speech transcription using the automatic speech recognition (asr) pipeline
def transcribe_speech(filepath): #path to the audio file
  if filepath is None: #if so, it displays a warning and return empty string  
    gr.Warning("No audio file found, please retry!")
    return ""
  output = asr(filepath) #invokes the asr pipeline on the audio file specified by filepath and  
  return output["text"] #returns the transcribed text from the output dictionary

# Initialize a Gradio interface for Microphone Transcribe
mic_transcribe = gr.Interface(
    fn = transcribe_speech, #specifies the function to be executed when the interface receives input
    inputs = gr.Audio(sources = "microphone",
                   type = "filepath"),
    outputs = gr.Textbox(label = "Transcription",
                         lines = 3),
    allow_flagging = "never" #specifies whether users are allowed to flag results or not
)

# Gradio interface for transcribing speech from an uploaded audio file.
file_transcribe = gr.Interface (
    fn = transcribe_speech,
    inputs = gr.Audio(sources = "upload",
                      type = "filepath"),
    outputs = gr.Textbox(label = "Transcription",
                         lines = 3),
    allow_flagging = "never"
)

# Create a tabbed interface using Gradio, that allows users to switch between two different interfaces:
# (mic_transcribe and file_transcribe) for transcribing speech
with demo:
  gr.TabbedInterface(
      [mic_transcribe,
       file_transcribe],
      ["Transcribe Microphone",
       "Transcribe Audio File"],
  )
demo.launch(debug = True )