from PIL import Image, ImageOps

async def generate_image(path:str) -> None:
    img = Image.open(f'uploaded/{path}')
    img_border = ImageOps.expand(img, border=100, fill='black')
    img_border.save(f'generated/{path}')