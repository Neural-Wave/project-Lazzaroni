



def get_image_path_based_on_transcription(transcription):
    image_map = {
        "happy": "images/happy.jpg",
        "default": "images/black.png"
    }

    for keyword, image_path in image_map.items():
        if keyword in transcription.lower():
            return image_path

    return image_map["default"]


def process_audio_and_text(audio):
    global step, dur
    transcription = transcriber.transcribe(audio)
-    image_path = get_image_path_based_on_transcription(transcription)
    reproduce_audio = None

    if transcriber.finished_talking:
        try:
            step += 1
            logger.info("Step: " + str(step))
            reproduce_audio = play_audio(step)
        except KeyError:
            pass
        transcriber.is_listening = False

    return transcription, image_path, step


with gr.Blocks() as demo:
    today = datetime.today()
    formatted_date = today.strftime("%d-%m-%Y %H:%M")

    gr.Markdown(
        f"""
        # 🎤 First job interview 
        ## Fullstack Software Engineer @ACME Inc.
        
        {formatted_date}
        """    
    )
    gr.Markdown("Please click on the Start Interview button when you are ready.")   
    with gr.Row():
        audio_output = gr.Interface(
            play_audio,
            [
                gr.Slider(0, 6, step=1),
            ],
            "audio",
        )

    with gr.Row():
        audio_input = gr.Audio(sources="microphone", streaming=True)
        
    output_text = gr.Textbox(label="Transcription")
    output_image = gr.Image(type="filepath", label="Generated Image")
    step_text = gr.Textbox("Interview step")
    
    audio_input.stream(
        process_audio_and_text,
        inputs=[audio_input],
        outputs=[output_text, output_image, step_text]
    )

demo.launch()
