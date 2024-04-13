#!python

import argparse
import logging
logger = logging.getLogger(__name__)
import pathlib
import sys
from urllib.request import urlretrieve
import webbrowser

import prompts
import embedding_management

from openai import OpenAI

def main():
    ## Arg Parse Section ##
    parser = argparse.ArgumentParser()

    # Subject options
    parser.add_argument("-a", "--native", help="Instead of prompting for a setting, tell GPT to describe the character in an 'appropriate setting'", action='store_true')

    # Model type options
    parser.add_argument("-g", "--gpt", help="GPT model version string, default 'gpt-4'", type=str, default="gpt-4", choices=['gpt-4', 'gpt-4-turbo-preview', 'gpt-3.5-turbo', 'gpt-3.5-turbo-instruct', 'babbage-002', 'davinci-002'])
    parser.add_argument("-d", "--dalle", help="DALL-E model version string, default 'dall-e-3'", type=str, default="dall-e-3", choices=['dall-e-3', 'dall-e-2'])
    

    # Image generation parameters
    parser.add_argument("-z", "--size", help="DALL-E image size string, default '1024x1024'. dall-e-2 only supports the default size.", type=str, default='1024x1024', choices=['1024x1024', '1024x1792', '1792x1024'])
    parser.add_argument("-q", "--quality", help="DALL-E image quality string, default 'standard'", type=str, default='standard', choices=['standard', 'hd'])
    parser.add_argument("-e", "--simple", help="Use the simplified final image prompt.", action="store_true")

    # Warning, n isn't well supported on the dall-e-3 api
    parser.add_argument("-m", "--img-num", help="DALL-E number of images to generate. Warning: Not supported by dall-e-3.", type=int, default=1, choices=range(1,11))
    parser.add_argument("-v", "--variation-num", help="GPT number of descriptions to generate and rank.", type=int, default=1, choices=range(1,11))

    # Output options
    parser.add_argument("-t", "--text", help="Only print the final image prompt, do not send it to DALL-E", action='store_true')
    parser.add_argument("-p", "--open", help="Automatically open all URL's returned by DALL-E", action='store_true')
    parser.add_argument("-i", "--interim", help="Print all the interim text results from GPT", action='store_true')
    parser.add_argument("-s", "--save", help="Save all the prompts and the generated image.", type=str, default=None)
    
    # Logging
    parser.add_argument("-l", "--log-level", help="Log level, 5: critical, 4: error, 3: warning, 2: info, 1: debug. Default: 5", type=int, choices=[1,2,3,4,5], default=5)
    parser.add_argument("-f", "--log-file",  help="A filename relative to CWD for the logs. If None logs are sent to stdout. Default: None", type=str, default=None)

    args = parser.parse_args()

    ## Logging and Save Setup ##
    if args.log_file is None:
        logging.basicConfig(stream=sys.stdout, level=10 * args.log_level)
    else:
        logging.basicConfig(filename=args.log_file, level=10 * args.log_level)

    if args.save:
        save_directory = pathlib.Path(args.save)
        save_directory.mkdir(parents=True, exist_ok=False)

    # Your API key must be saved in an env variable for this to work.
    client = OpenAI()

    ## Collect Input Section ## 
    character_name = input("Character: ")
    name_replacement = input("Name replacement: ")

    image_setting = None
    if not args.native:
        image_setting = input("Setting: ")
    logger.debug("Setting stored as %s", image_setting)

    image_style = input("Style: ")
    logger.debug("Style stored as %s", image_style)

    logger.debug("Character stored as %s to be replaced with %s", character_name, name_replacement)
    text_to_save = f'Character: {character_name}\nReplacement: {name_replacement}\nSetting:{image_setting}\nStyle: {image_style}\n\n'

    ## Begin Prompting Section ##
    ## Character Details ##
    image_subjects = prompts.fetch_character_description(client, args.gpt, character_name, args.variation_num)

    if args.variation_num > 1:
        subject_descriptions = [choice.message.content for choice in image_subjects]
        
        # TODO: perhaps this magic string is bad idea.
        all_choices = embedding_management.sort_by_nearest(client, 'text-embedding-3-large', prompts.CHARACTER_PROMPT.format(character=character_name), subject_descriptions)
        text_to_save += 'Subject Descriptions ranked by embedding distance:\n\n'
        if args.interim: print('Subject Descriptions ranked by embedding distance:\n\n')
        
        for choice in all_choices:
            text_to_save += f'  {choice[1]}: {choice[0]}\n\n'
            if args.interim: print(f'  {choice[1]}: {choice[0]}\n\n')
        
        image_subject = all_choices[0][0]
    else:
        image_subject = image_subjects[0].message.content

    text_to_save += f'Expanded Character Details:\n{image_subject}\n\n'
    if args.interim:
        print(f'Expanded Character Details:\n{image_subject}\n\n')
    
    # Sanitize the output to avoid giving the name of the character to the image generator
    # First all full copies with the replacement
    image_subject = image_subject.replace(character_name, name_replacement)

    # Then any lingering first or last names alone
    for name_component in character_name.split(' '):
        image_subject = image_subject.replace(name_component, name_replacement)

    text_to_save += f'Scrubbed Character Details:\n{image_subject}\n\n'
    if args.interim:
        print(f'Scrubbed Character Details:\n{image_subject}\n\n')
    

    ## Combine Setting With Character ##
    content_details = prompts.fetch_scene_details(client, args.gpt, image_subject, image_setting)
    text_to_save += f'Content Detail:\n{content_details}\n\n'
    if args.interim:
        print(f'Content details:\n{content_details}\n\n')


    ## Fetch Style Details ##
    style_details = prompts.fetch_style_detail(client, args.gpt, image_style)
    text_to_save += f'Style details:\n{style_details}\n\n'
    if args.interim:
        print(f'Style details:\n{style_details}\n\n')

    ## Image Generation Section ##
    if args.simple:
        image_prompt = prompts.SIMPLIFIED_IMG_PROMPT.format(scene_details=content_details, style=style_details)
    else:
        image_prompt = prompts.fetch_dalle_prompt(client, args.gpt, content_details, style_details)
    
    text_to_save += f'Final prompt:\n{image_prompt}\n\n'

    # dall-e-2 has a 1000 character max for prompt, dall-e-3 is 4000 characters
    # TODO: handle this more gracefully?
    if args.dalle == 'dall-e-2' and len(image_prompt) > 1000:
        logger.warning('Prompt was too long for dall-e-2, clipping')
        image_prompt = image_prompt[:1000]
    elif args.dalle == 'dall-e-3' and len(image_prompt) > 4000:
        logger.warning('Prompt was too long for dall-e-3, clipping')
        image_prompt = image_prompt[:4000]
    
    print(f'Final prompt: \n{image_prompt}\n')

    if not args.text:
        # The I NEED thing is from OpenAI docs, to reduce prompt rewriting.
        img_response = client.images.generate(
            model=args.dalle,
            prompt=f'I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: {image_prompt}',
            size=args.size,
            quality=args.quality,
            n=args.img_num
        )

        for idx, img_data in enumerate(img_response.data):
            if img_data.revised_prompt:
                rewritten_output = f'Prompt rewritten by OpenAI: \n\n {img_data.revised_prompt}\n\n'
                print(rewritten_output)
                text_to_save += rewritten_output

            print(img_data.url)
            text_to_save += f'{img_data.url} \n\n'

            if args.open:
                webbrowser.open_new_tab(img_data.url)
            
            if args.save:
                img_save_path = save_directory / f'{idx}.png' # TODO: more robust
                urlretrieve(img_data.url, img_save_path)

    if args.save:
        save_text_path = save_directory / "prompts.txt"
        with save_text_path.open("w", encoding ="utf-8") as f:
            f.write(text_to_save)



if __name__ == '__main__':
    main()