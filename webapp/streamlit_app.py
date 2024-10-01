import streamlit as st
import json
from lime import lime_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.request
from urllib.error import HTTPError
from functools import partial
from io import BytesIO
import zipfile
import os

WEBAPP_DATASET_PATH = "../data"
IMAGENET_PATH = os.path.join(WEBAPP_DATASET_PATH, "TinyImageNet/")

def set_device():
    device = None

    # Fetching the device that will be used throughout this notebook
    if torch.backends.mps.is_available():
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.use_deterministic_algorithms(True)

        #set device to use mps
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #set device to use cuda
        device = torch.device("cuda:0")
    else:
        #set device to use the cpu
        device = torch.device("cpu")

    print("Using device", device)
    return device

def load_dataset():
    # Github URL where the dataset is stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial10/"

    os.makedirs(WEBAPP_DATASET_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    file_name = "TinyImageNet.zip"
    file_path = os.path.join(WEBAPP_DATASET_PATH, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)
        if file_name.endswith(".zip"):
            print("Unzipping file...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(file_path.rsplit("/",1)[0])
                print("Unzip complete")
    
    # Load dataset and create data loader
    assert os.path.isdir(IMAGENET_PATH), f"Could not find the ImageNet dataset at expected path \"{IMAGENET_PATH}\". " + \
                                        f"Please make sure to have downloaded the ImageNet dataset here, or change the {WEBAPP_DATASET_PATH=} variable."

def get_label_names_and_folders():
    # Load label names to interpret the label numbers 0 to 999
    with open(os.path.join(IMAGENET_PATH, "label_list.json"), "r") as f:
        label_names = json.load(f)

    # get a list of folders in sorted order for retrieving pictures by label
    folders = sorted([f for f in os.listdir(IMAGENET_PATH) if os.path.isdir(os.path.join(IMAGENET_PATH, f))])

    return folders, label_names

def load_model(device):
    # Load a pre-trained ResNet-34 model with ImageNet weights
    pretrained_model = models.resnet34(weights='IMAGENET1K_V1')
    pretrained_model = pretrained_model.to(device)

    # Set model to evaluation mode
    pretrained_model.eval()

    # Specify that no gradients needed for the network
    for p in pretrained_model.parameters():
        p.requires_grad = False
    
    return pretrained_model

def get_images(label_name, label_list, folders):
    '''
    gets a list of images in RGB format by label name from the TinyImageNet dataset

    Inputs:
        label_name (str): the label for which to retrieve the images

    Return:
        images (list): a list of the images retrieved
    '''
    #get the index of the label from label_list.json
    index = label_list.index(label_name)

    #get the corresponding folder of images from TinyImageNet
    folder = IMAGENET_PATH + folders[index] + '/'

    #get the images from the selected folder
    image_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    images = []
    for image_name in image_names:
        #open the image
        with open(os.path.relpath(folder + image_name), 'rb') as f:
            with Image.open(f) as img:
                #convert the image to RGB and add it to the output list
                images.append(img.convert('RGB'))

    return images

def get_plain_transform():
    # Convert the input image to PyTorch Tensor, normalize the images using the mean and standard deviation above and
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]),
                                std=np.array([0.229, 0.224, 0.225]))
    ])

def get_pil_transform():
    # Resize and take the center part of image to what our model expects
    return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224)
        ])

def batch_predict(image, device, model):
    '''
    Add a batch dimension to a Pytorch tensor, get predictions from the model, and return the probabilities

    Input:
        image (Pytorch Tensor): the image used for prediction

    Return:
        A numpy array of the computed probabilities
    '''
    plain_transform = get_plain_transform()

    # apply transformations to each image and stack them into a batch (a tensor) along a new dimension.
    batch = torch.stack(tuple(plain_transform(i) for i in image), dim=0)

    # move the batch to the device
    batch = batch.to(device)

    # feed the batch to the model to get the logits
    logits = model(batch)

    # convert logits to probabilities
    probs = F.softmax(logits, dim=1)

    # detach the computed probabilities from the computational graph, move them back to the CPU and convert them into a numpy array
    return probs.detach().cpu().numpy()

def get_explanations(image_list, device, model):
    # instantiate the LimeImageExplainer and explanations array
    explainer = lime_image.LimeImageExplainer()
    explanations = []

    predict_fn = partial(batch_predict, device=device, model=model)

    pil_transform = get_pil_transform()

    progress_bar = st.progress(0)

    # get explanations for each image
    for i, img in enumerate(image_list):
        explanations.append(explainer.explain_instance(np.array(pil_transform(img)),
                                                predict_fn, # classification function
                                                top_labels=5,
                                                hide_color=0,
                                                num_samples=1000))
        
        progress_bar.progress(int((i + 1) / len(image_list) * 100))
    
    progress_bar.empty()
    
    return explanations

def plot_explanation(explanation, col):
    # get the image and mask for each image
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)

    # normalize the image to range [0, 1]
    temp = temp.astype(float) / 255.0

    # convert mask to bool to apply changes to the unimportant part of the image
    mask = mask.astype(bool)

    # Create a gray image to replace the unimportant parts
    gray_image = np.ones_like(temp) * 0.5

    # Combine the important parts of the original image and the gray image for the unimportant parts
    temp[~mask] = gray_image[~mask]

    # Add boundaries to the image
    img_boundry2 = mark_boundaries(temp, mask)
    
    # Create a smaller figure for each image
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figsize to control the size
    ax.imshow(img_boundry2)
    ax.axis("off")  # Turn off the axis

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")  # Save to buffer
    buf.seek(0)  # Rewind the buffer to the beginning

    # Use st.image to display the image (no expand button)
    col.image(buf, use_column_width=True)

    # Close the figure to free up memory
    plt.close(fig)

# Main function that creates the streamlit web app
def main():
    if 'initialized' not in st.session_state:
        # Initialization block - only runs once
        st.session_state['initialized'] = True

        # download the dataset
        load_dataset()

    if 'device' not in st.session_state:
        # set the device to use
        st.session_state['device'] = set_device()

    # Assign the device from session state
    device = st.session_state['device']
    
    if 'model' not in st.session_state:
        # load the pretrained model
        st.session_state['model'] = load_model(device)

    # Assign the model from session state
    model = st.session_state['model']

    st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")

    title = """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    <h1 class="centered-title">Resnet34 LIME Visualizer</h1>
    """

    #Setting the web page's title 
    st.markdown(title, unsafe_allow_html=True)

    sidebar_footer = """
    <style>
    .sidebar-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 10%;
        background-color: rgb(240, 242, 246);
        color: black;
        text-align: center;
    }
    </style>
    <div class="sidebar-footer">
        <p>Author: Tal Erez</p>
    </div>
    """
    st.sidebar.markdown(sidebar_footer, unsafe_allow_html=True)

    # Create a sidebar to select the pre-built dataset to plot or file upload
    st.sidebar.header('Image Options')

    # get folders and label names
    folders, label_list = get_label_names_and_folders()

    # Create a dropdown (selectbox) for the string list
    selected_label = st.sidebar.selectbox('Select a label', ['(None)'] + label_list)
    st.sidebar.markdown("""
    <div>
        <h2>Description</h2>
        <p>This site demonstrates how to use LIME to generate local explanations for images. The images used come from a small version of the ImageNet library, and the pretrained model used for predictions is ResNet34. For a given label, this will output LIME's explanations for each image of that label in the dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if selected_label == '(None)':
        info = """
        <style>
        .centered-title {
            text-align: center;
        }
        </style>
        <h2 class="centered-title"><== Select a label to generate explanations</h2>
        """

        #Setting the web page's title 
        st.markdown(info, unsafe_allow_html=True)
    else:
        # get list of images
        images = get_images(selected_label, label_list, folders)

        st.text("Original Images")

        cols = st.columns(len(images))  # Create two columns to display images side by side

        hide_img_fs = '''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        '''

        st.markdown(hide_img_fs, unsafe_allow_html=True)

        st.markdown(
            """
            <style>
            .main .block-container {
                max-width: 1200px;  /* Adjust the width as needed */
                padding-left: 2rem;
                padding-right: 2rem;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )

        for i, img in enumerate(images):
            # Select the appropriate column to display the image
            col = cols[i]
            
            # Create a smaller figure for each image
            fig, ax = plt.subplots(figsize=(20,20))  # Adjust figsize to control the size
            ax.imshow(img)
            ax.axis("off")  # Turn off the axis

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")  # Save to buffer
            buf.seek(0)  # Rewind the buffer to the beginning

            # Use st.image to display the image (no expand button)
            col.image(buf, use_column_width=True)

            # Close the figure to free up memory
            plt.close(fig)

        lime_text_placeholder = st.empty()
        cols_lime = st.columns(len(images))
        placeholders = [col.empty() for col in cols_lime]

        # Display loading spinner while processing
        with st.spinner('Generating explanations...'):
            explanations = get_explanations(images, device, model)

        lime_text_placeholder.text("Most Important Features For Prediction as Detected by LIME")

        for i, explanation in enumerate(explanations):
            # Select the appropriate column to display the explanation
            col = placeholders[i]
            
            # Plot the explanation in the column (replace this with actual plotting code)
            plot_explanation(explanation, col)
    
    st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1200px;  /* Adjust the width as needed */
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
    )  
    
if __name__ == '__main__':
    main()