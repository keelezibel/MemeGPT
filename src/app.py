import gradio as gr
import os
import whisper
from pytube import YouTube
from yt_dlp import YoutubeDL

class GradioInference():
    def __init__(self):
        self.sizes = list(whisper._MODELS.keys())
        self.langs = ["none"] + sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.current_size = "medium"
        self.loaded_model = whisper.load_model(self.current_size)
        self.yt = None
    
    def download_videos(self, link):
        """Specify the yt-dlp parameters
        Args:
            url (str): URL to retrieve videl
            name (str): speaker name
        """
        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "postprocessors": [
                {  # Extract audio using ffmpeg
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "outtmpl": f"{os.path.curdir}/tmp.%(ext)s",
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download(link)

        return f"{os.path.curdir}/tmp.wav"

    def detect_lang(self, path):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(path)
        audio_segment = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio_segment).to(self.loaded_model.device)

        # detect the spoken language
        _, probs = self.loaded_model.detect_language(mel)
        language = max(probs, key=probs.get)

        return language

    def __call__(self, link, lang, size, subs):
        if self.yt is None:
            self.yt = YouTube(link)
        path = self.download_videos(link)

        if size != self.current_size:
            self.loaded_model = whisper.load_model(size)
            self.current_size = size
        
        if lang == "none":
            lang = self.detect_lang(path)

        options = whisper.DecodingOptions().__dict__.copy()
        options["language"] = lang
        options["beam_size"] = 5
        options["best_of"] = 5
        del options["task"]
        transcribe_options = dict(task="transcribe", **options)
        translate_options = dict(task="translate", **options)
        results = self.loaded_model.transcribe(path, **transcribe_options)
        translation_txt = self.loaded_model.transcribe(path, **translate_options)["text"]

        if subs == "None":
            return results["text"], translation_txt
        elif subs == ".srt":
            return self.srt(results["segments"]), translation_txt
        elif ".csv" == ".csv":
            return self.csv(results["segments"]), translation_txt
   
    def srt(self, segments):
        output = ""
        for i, segment in enumerate(segments):
            output += f"{i+1}\n"
            output += f"{self.format_time(segment['start'])} --> {self.format_time(segment['end'])}\n"
            output += f"{segment['text']}\n\n"
        return output
  
    def csv(self, segments):
        output = ""
        for segment in segments:
            output += f"{segment['start']},{segment['end']},{segment['text']}\n"
        return output

    def format_time(self, time):
        hours = time//3600
        minutes = (time - hours*3600)//60
        seconds = time - hours*3600 - minutes*60
        milliseconds = (time - int(time))*1000
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"
    
    def populate_metadata(self, link):
        self.yt = YouTube(link)
        return self.yt.thumbnail_url, self.yt.title

gio = GradioInference()
title="Youtube Whisperer"
description="Speech to text transcription of Youtube videos using OpenAI's Whisper"

block = gr.Blocks()
with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 500px; margin: 0 auto;">
              <div>
                <h1>Youtube Whisperer</h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Speech to text transcription of Youtube videos using OpenAI's Whisper
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                sz = gr.Dropdown(label="Model Size", choices=gio.sizes, value='base')
                lang = gr.Dropdown(label="Language (Optional)", choices=gio.langs, value="none")
            with gr.Row().style(equal_height=True):
                wt = gr.Radio(["None", ".srt", ".csv"], label="With Timestamps?")
            link = gr.Textbox(label="YouTube Link")
            title = gr.Label(label="Video Title")
            with gr.Row().style(equal_height=True):
                img = gr.Image(label="Thumbnail")
                transcript = gr.Textbox(label="Transcription", placeholder="Transcription Output", lines=10)
                translate = gr.Textbox(label="Translation", placeholder="Translation Output", lines=10)
            with gr.Row().style(equal_height=True): 
                btn = gr.Button("Transcribe")       
            btn.click(gio, inputs=[link, lang, sz, wt], outputs=[transcript, translate])
            link.change(gio.populate_metadata, inputs=[link], outputs=[img, title])
block.launch()