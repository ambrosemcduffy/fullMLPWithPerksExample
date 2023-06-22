from bing_image_downloader import downloader





# Define the search query and number of images to download
limit = 30
query = "Metal Gear "
downloader.download(query, limit=limit, output_dir='images', adult_filter_off=True, force_replace=False, timeout=60)
