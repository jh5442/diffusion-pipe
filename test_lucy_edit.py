from typing import List

import torch
from PIL import Image

from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import export_to_video, load_video


# Arguments
data = [
    {"video_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/data/pexel_lucy_edit/test_01.mp4",
     "prompt":"Change the background, make this guy in a coffee shop. Make sure the main person, his clothing, accessories unchanged",
     "save_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/results/lucy/test_01_indoor.mp4",
     "width" : 720 ,
     "height" : 1280,},

    {"video_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/data/pexel_lucy_edit/test_01.mp4",
     "prompt":"change the weather, make it rain heavily. The person's hair and clothes get wet",
     "save_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/results/lucy/test_01_heavy_rain.mp4",
     "width": 720,
     "height": 1280,},

    {"video_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/data/pexel_lucy_edit/test_02.mp4",
     "prompt":"Make it heavy rain. We can see the two people and their dogs get wet, water dripping from their hair",
     "save_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/results/lucy/test_02_heavy_rain.mp4",
     "width": 1280,
     "height": 720,
     },

    {"video_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/data/pexel_lucy_edit/test_02.mp4",
     "prompt":"Make it heavy snow, we can see snow accumulating on these people and the dogs",
     "save_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/results/lucy/test_02_heavy_snow.mp4",
     "width": 1280,
     "height": 720,
     },

    {"video_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/data/pexel_lucy_edit/test_03.mp4",
     "prompt":"dramatical disco lighting",
     "save_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/results/lucy/test_03_disco.mp4",
     "width": 1280,
     "height": 720,
     },

    {"video_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/data/pexel_lucy_edit/test_03.mp4",
     "prompt":"Make the left side of her (including face and body) lighten up, the right side in darkness",
     "save_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/results/lucy/test_03_half_light.mp4",
     "width": 1280,
     "height": 720,
     },

    {"video_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/data/pexel_lucy_edit/test_04.mp4",
     "prompt":"First, turn all the lights off. Then apply a focused beam of light that highlights the person’s face, casting hard shadow",
     "save_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/results/lucy/test_04_half.mp4",
     "width": 1280,
     "height": 720,
     },

    {"video_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/data/pexel_lucy_edit/test_04.mp4",
     "prompt":"Cinematic lighting",
     "save_path":"/picassox/intelligent-cpfs/pixocial/jin.huang/results/lucy/test_04_cinematic.mp4",
     "width": 1280,
     "height": 720,
     },
]


# Load video
# def convert_video(path) -> List[Image.Image]:
#     video = load_video(path)[:num_frames]
#     video = [video[i].resize((width, height)) for i in range(num_frames)]
#     return video

num_frames = 81

# Load model
model_id = "decart-ai/Lucy-Edit-Dev"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = LucyEditPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

for one_data_dict in data:
    negative_prompt = ""

    print("Processing video {}".format(one_data_dict["video_path"]))
    video = load_video(one_data_dict["video_path"])[:num_frames]
    video = [video[i].resize((one_data_dict["width"], one_data_dict["height"])) for i in range(num_frames)]

    prompt = one_data_dict["prompt"]

    # Generate video
    # output = pipe(
    #     prompt=prompt,
    #     video=video,
    #     negative_prompt=negative_prompt,
    #     height=one_data_dict["height"],
    #     width=one_data_dict["width"],
    #     num_frames=81,
    #     guidance_scale=5.0
    # ).frames[0]
    output = pipe(
        prompt=prompt,
        video=video,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0
    ).frames[0]

    # Export video
    export_to_video(output, one_data_dict["save_path"], fps=16)