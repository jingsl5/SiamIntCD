import torch

from flask import Flask, request, jsonify
import numpy as np
import requests
from PIL import Image
from diffusers import DiffusionPipeline
import base64
import json
import cv2
from PIL import Image

app = Flask(__name__)

imgformat = ['png','jpg','jpeg','tiff','gif','tif','bmp']

generated_image_base64 = None

class StableDiffusionInference:
    
    pipe = DiffusionPipeline.from_pretrained("/disk3/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")


    def process_text(self, text):
        image = self.pipe(
                text,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
            ).images[0]
    
        img_base64, code = self.convert_to_base64(image)
        
        return img_base64, code

    def convert_to_base64(self, result):
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(np.array(result),cv2.COLOR_BGR2RGB))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64, 0
 



inference = StableDiffusionInference()


@app.route('/sdxl', methods=['POST'])
def process_image_route():
    global generated_image_base64
    try:
        if request.method == 'POST':
            user_input = request.form.get('text')
            if user_input is None or 'xijinping' in user_input or 'jinping xi' in user_input:
                return jsonify({'error': 'No text provided in POST request'}), 400

            # 生成图像
            img_base64, code = inference.process_text(user_input)

            # 存储生成的图像
            generated_image_base64 = img_base64

            return {'images':generated_image_base64}

            # # 返回包含生成图像的 HTML 页面（仅显示图像）
            # return f'''
            # <!DOCTYPE html>
            # <html lang="en">
            # <head>
            #     <meta charset="UTF-8">
            #     <meta name="viewport" content="width=device-width, initial-scale=1.0">
            #     <title>Generated Image</title>
            #     <style>
            #         body {{
            #             margin: 0;
            #             display: flex;
            #             justify-content: center;
            #             align-items: center;
            #             height: 100vh;
            #             background-color: #000;
            #         }}
            #         img {{
            #             max-width: 100%;
            #             max-height: 100%;
            #         }}
            #     </style>
            # </head>
            # <body>
            #     <img src="data:image/jpeg;base64,{img_base64}" alt="Generated Image">
            # </body>
            # </html>
            # '''

    except Exception as e:
        print('=====')
        print(str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/sdxl', methods=['GET'])
def display_image_route():
    # global generated_image_base64
    if generated_image_base64:
        return {'images':generated_image_base64}
        # return f'''
        # <!DOCTYPE html>
        # <html lang="en">
        # <head>
        #     <meta charset="UTF-8">
        #     <meta name="viewport" content="width=device-width, initial-scale=1.0">
        #     <title>Generated Image</title>
        #     <style>
        #         body {{
        #             margin: 0;
        #             display: flex;
        #             justify-content: center;
        #             align-items: center;
        #             height: 100vh;
        #             background-color: #000;
        #         }}
        #         img {{
        #             max-width: 100%;
        #             max-height: 100%;
        #         }}
        #     </style>
        # </head>
        # <body>
        #     <img src="data:image/jpeg;base64,{generated_image_base64}" alt="Generated Image">
        # </body>
        # </html>
        # '''
    else:
        return {'images':generated_image_base64}
        # return '''
        # <!DOCTYPE html>
        # <html lang="en">
        # <head>
        #     <meta charset="UTF-8">
        #     <meta name="viewport" content="width=device-width, initial-scale=1.0">
        #     <title>Image Generator</title>
        # </head>
        # <body>
        #     <h1>No image generated yet. Please send a POST request with the text parameter to generate an image.</h1>
        # </body>
        # </html>
        # '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8012)#, debug=True)



