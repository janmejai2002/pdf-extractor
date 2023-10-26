import os
import cv2
import numpy as np
import fitz
from PIL import Image
import img2pdf
import argparse
from tqdm.auto import tqdm
import shutil
from flask import Flask, request, render_template, send_file, send_from_directory, url_for

app = Flask(__name__)

# Function to get the maximum image dimensions
def get_max_image_dimensions(folder_path):
    max_width = 0
    max_height = 0

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            if image is not None:
                height, width, _ = image.shape
                max_width = max(max_width, width)
                max_height = max(max_height, height)

    return max_width

# Function to stitch images
def stitch_all(folder_path, image_list):
    images = []
    output_width = get_max_image_dimensions(folder_path=folder_path)

    for filename in image_list:
        image = cv2.imread(filename)

        if image is not None:
            current_width = image.shape[1]
            border_width = output_width - current_width
            left_border_width = border_width // 2
            right_border_width = border_width - left_border_width

            canvas = np.ones((image.shape[0], output_width, 3), np.uint8) * 255
            canvas[:, left_border_width:left_border_width + current_width, :] = image

            images.append(canvas)

    combined_image = np.vstack(images)
    return combined_image


def sheet(input_image_path, output_dir):
    # Load the input image using cv2
    output_dir = "output_images"  # Directory to save split images

    image = input_image_path
    split_height = 1555  # Height at which to split the image

    if image is None:
        print("Error: Image not found.")
        return

    height, width, channels = image.shape

    # Check if the split height is within the image's height bounds
    if split_height <= 0 or split_height >= height:
        print("Error: Invalid split height.")
        return

    # Determine the number of splits
    num_splits = height // split_height

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the image and save each part
    imgs = []
    start = 0
    split_count = 0

    while start < height:
        end = min(start + split_height, height)
        split_image = image[start:end, :]
        # Save the split image
        output_path = os.path.join(output_dir, f"split_{split_count}.png")
        cv2.imwrite(output_path, split_image)

        # print(f"Saved: {output_path}")
        imgs.append(output_path)
        split_count += 1
        start = end



    return imgs


def resize_img(image):
    r = 1080 / image.shape[1]
    dim = (1080 ,int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized
# Define a function to remove consecutive rows of all-white pixels
def remove_consecutive_white_rows(image):
    result = []
    white_row_count = 0
    max_consecutive_white_rows = 5  # Adjust as needed

    for row in image:
        if np.all(row == 255):  # Check if the row is all-white
            white_row_count += 1
        else:
            white_row_count = 0

        if white_row_count <= max_consecutive_white_rows:
            result.append(row)

    return np.array(result)

def clean_loop(image):
    while True:
        new_image = remove_consecutive_white_rows(image)
        if np.array_equal(new_image, image):
            break  # No more consecutive white rows to remove
        image = new_image

    return image

def process_image(image):
#Goodnotes Green
    # lower_green = (0, 165, 0)
    # upper_green = (215, 255, 168)
    lower_green = (82, 225, 245)
    upper_green = (150, 250, 255)
    green_mask = cv2.inRange(image, lower_green, upper_green)
    # cv2.imwrite('mask.jpg', green_mask)
    kernel = np.ones((3, 3), np.uint8)
    openning = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.dilate(openning, kernel, iterations=3)


    preserved_highlights = cv2.bitwise_and(image, image, mask=closing)
    # cv2.imwrite('mask1.jpg',preserved_highlights )
    gray = cv2.cvtColor(preserved_highlights, cv2.COLOR_RGB2GRAY)
    _, preserved_text = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)
    preserved_text = cv2.bitwise_and(preserved_text, preserved_text, mask=closing)

    final_image = 255 - preserved_text
    cleaned_image = clean_loop(final_image)
    
    # cv2.imwrite('clean.jpg', cleaned_image)
    resized_img=resize_img(cleaned_image)
    cv2.imwrite('final.jpg', resized_img)
    return resized_img      

def main(input_pdf):
    name = os.path.splitext(os.path.basename(input_pdf))[0]
    output_dir = os.path.join('output', name)  # Specify the output directory

    os.makedirs(output_dir, exist_ok=True)

    pdf_document = fitz.open(input_pdf)
    processed_image_paths = []

    for page_number in tqdm(range(pdf_document.page_count)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)
        image_np = np.array(image)

        # Convert BGR to RGB format (if needed)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        cv2.imwrite('input.jpg', image_rgb)
        # print(image.mode)
        processed_image = process_image(image_rgb)


        if processed_image is not None and processed_image.shape[0] > 3:
            image_path = os.path.join(output_dir, f'page_{page_number + 1}.jpg')
            cv2.imwrite(image_path, processed_image)
            processed_image_paths.append(image_path)
        else:
            print(f"Skipping page {page_number + 1} due to processing issues.")

    pdf_document.close()
    if not processed_image_paths:
        print("No valid images were processed.")
        return None, None

    combined_image = stitch_all(output_dir, processed_image_paths)

    output_pdf = os.path.join('output', f'{name}_final.pdf')  # Specify the output PDF path

    sheets = sheet(combined_image, output_pdf)

    with open(output_pdf, 'wb') as pdf_output:
        pdf_output.write(img2pdf.convert(sheets))

    return name, output_pdf  # Return the path to the generated PDF


@app.route('/')
def index():
    return "Welcome to the PDF Processing Web App. <a href='/upload'>Upload a PDF</a>"

# Create a route for uploading and processing PDFs
@app.route('/upload', methods=['GET', 'POST'])
def upload_pdf():
    download_link = None  # Initialize as None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            input_pdf_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(input_pdf_path)
            name, output_pdf_path = main(input_pdf_path)  # Process the uploaded PDF
            normalized_output_path = os.path.normpath(output_pdf_path)
            download_link = '/download/' + os.path.basename(output_pdf_path)

            print(f"DOWNLOAD LINK - {download_link}")

    return render_template('upload.html', download_link=download_link)

@app.route('/download/<path:pdf_filename>')
def download_pdf(pdf_filename):
    # Specify the directory where the processed PDFs are stored
    output_dir = 'output'
    try:
        return send_from_directory(output_dir, pdf_filename, as_attachment=True)
    except Exception as e:
        return str(e)  # Print any error that occurs during file retrieval

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    if not os.path.exists('output'):
        os.mkdir('output')
    app.run()
