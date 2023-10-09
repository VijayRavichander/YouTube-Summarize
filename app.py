from flask import Flask, request, render_template
import openai
import numpy as np
import config
import whisper
import openai
from pytube import YouTube

app = Flask(__name__)

openai.api_key = config.OPENAI_API_KEY
model = whisper.load_model('base')

@app.route('/static/<path:filename>')
def serve_static(filename):
  return app.send_static_file(filename)

@app.route('/')
def search_form():
  return render_template('search_form.html')


@app.route('/search')
def search():
    # Get the search query from the URL query string
    query = request.args.get('query')
    youtube_video_url = str(query)
    youtube_video = YouTube(youtube_video_url)

    # Getting the video title
    video_title = youtube_video.title

    # Downloading the audio
    streams = youtube_video.streams.filter(only_audio=True)
    stream = streams.first()
    stream.download(filename='mkbhd_audio.mp4')

    transcribed_output = model.transcribe("mkbhd_audio.mp4")
    transcript = transcribed_output['text']

    # Chunking the transcript
    words = transcript.split(" ")
    n_chunks = int( len(words) / 600)
    chunks = np.array_split(words, n_chunks)


    # Summary for all the chunks
    summary_responses = []

    for chunk in chunks:

        sentences = ' '.join(list(chunk))

        prompt = f"{sentences}\n\ntl;dr:"

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.3, # The temperature controls the randomness of the response, represented as a range from 0 to 1. A lower value of temperature means the API will respond with the first thing that the model sees; a higher value means the model evaluates possible responses that could fit into the context before spitting out the result.
            max_tokens=150,
            top_p=1, # Top P controls how many random results the model should consider for completion, as suggested by the temperature dial, thus determining the scope of randomness. Top P’s range is from 0 to 1. A lower value limits creativity, while a higher value expands its horizons.
            frequency_penalty=0,
            presence_penalty=1
        )

        response_text = response["choices"][0]["text"]
        summary_responses.append(response_text)

    results = "".join(summary_responses)

    # Render the search results template, passing in the search query and results
    return render_template('search_results.html', query=query, results=results)

if __name__ == '__main__':
  app.run()