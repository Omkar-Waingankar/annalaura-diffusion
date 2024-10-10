import anthropic
import base64
import os
import json

client = anthropic.Anthropic()

MEDIA_TYPE = "image/jpeg"
EXAMPLE_IMAGE_1 = "data/C0KkPzbRpdR_0.jpg"
EXAMPLE_PROMPT_1 = "two small blue elephants standing on rolling green hills and touching their trunks"
EXAMPLE_IMAGE_2 = "data/C0KkPzbRpdR_1.jpg"
EXAMPLE_PROMPT_2 = "small monkey sitting and reading blue book against a tree in park"
EXAMPLE_IMAGE_3 = "data/C0cKm1WR5Wq_1.jpg"
EXAMPLE_PROMPT_3 = "orange cat using phone to take pictures of bugs in a garden"

def fetch_and_encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def filename_to_prompt_messages(filename):
    return [
        {
            "type": "text",
            "text": f"This drawing's filename is: {filename}"
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": MEDIA_TYPE,
                "data": fetch_and_encode_image(f"data/{filename}")
            }
        }
    ]

PROMPT = "Please generate a prompt for the drawings below. The prompt should be all lowercase. Don't use the word 'drawing' or 'cartoon' in your prompt. If there's text in the drawing, don't include that in your prompt."

FEW_SHOT_MESSAGES = [
    {
        "type": "text",
        "text": "I need your help generating prompts for a bunch of drawings to train a text2img model. Please generate a prompt for each drawing that describes the cartoon characters and/or scenery of the drawing in a short sentence. Below, I will give you three example images and their expected prompts to help guide you."
    },
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": MEDIA_TYPE,
            "data": fetch_and_encode_image(EXAMPLE_IMAGE_1)
        }
    },
    {
        "type": "text",
        "text": "Expected prompt: " + EXAMPLE_PROMPT_1
    },
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": MEDIA_TYPE,
            "data": fetch_and_encode_image(EXAMPLE_IMAGE_2)
        }
    },
    {
        "type": "text",
        "text": "Expected prompt: " + EXAMPLE_PROMPT_2
    },
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": MEDIA_TYPE,
            "data": fetch_and_encode_image(EXAMPLE_IMAGE_3)
        }
    },
    {
        "type": "text",
        "text": "Expected prompt: " + EXAMPLE_PROMPT_3
    },
]

def generate_and_save_labels(start_index, end_index):
    prompt_messages = [
        {
            "type": "text",
            "text": PROMPT
        }
    ]

    img_fnames = []
    for f in os.listdir("data"):
        if f.endswith(".jpg"):
            img_fnames.append(f)
    img_fnames = sorted(img_fnames)

    for img_fname in img_fnames[start_index:end_index]:
        prompt_messages.extend(filename_to_prompt_messages(img_fname))

    full_content = []
    full_content.extend(FEW_SHOT_MESSAGES)
    full_content.extend(prompt_messages)

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a world-class data annotator for drawings.",
        messages=[
            {
                "role": "user",
                "content": full_content
            }
        ],
        tools=[
            {
                "name": "json_formatter",
                "description": "Formats the output as an array of structured JSON objects with 'file_name' and 'prompt' keys.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_name": {"type": "string"},
                                    "prompt": {"type": "string"}
                                },
                                "required": ["file_name", "prompt"]
                            }
                        }
                    }
                }
            }
        ],
        tool_choice={"type": "tool", "name": "json_formatter"}
    )

    for content_block in message.content:
        f = open("labeler_outputs/dump_{}_{}.json".format(start_index, end_index), "w+")
        f.write(json.dumps(content_block.input, indent=2))
        f.close()

for i in range(640, 860, 10):
    generate_and_save_labels(i, i + 10)