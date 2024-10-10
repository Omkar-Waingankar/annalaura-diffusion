import instaloader
from instaloader import Post
from datetime import datetime
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Scrape Instagram post images by shortcode.')
parser.add_argument('--shortcode', type=str, help='The shortcode of the Instagram post to scrape.')
args = parser.parse_args()

bot = instaloader.Instaloader()

shortcode = args.shortcode
post = Post.from_shortcode(bot.context, shortcode)
sidecar_nodes = post.get_sidecar_nodes()
sidecar_nodes = [node for node in sidecar_nodes if node.is_video == False]

if len(sidecar_nodes) == 0:
    filename = "scraped_images/{}".format(shortcode)
    bot.download_pic(filename, post.url, datetime.now())
else:
    for i, sidecar in enumerate(sidecar_nodes):
        filename = "scraped_images/{}_{}".format(shortcode, str(i))
        bot.download_pic(filename, sidecar.display_url, datetime.now())
