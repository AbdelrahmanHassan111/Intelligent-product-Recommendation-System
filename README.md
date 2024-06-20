# E-Commerce Product Classification and Visualization 📊🛍️

## Project Overview

The growing e-commerce industry presents a large dataset waiting to be scraped and researched upon. This dataset includes professionally shot high-resolution product images, multiple label attributes describing each product, and descriptive text that comments on the product characteristics.

## Dataset Description 📁

Each product is identified by an ID like `42431`. The dataset consists of the following components:
- **styles.csv**: Contains the mapping of product IDs to their attributes.
- **images/**: Contains product images, e.g., `images/42431.jpg`.

### Content of `styles.csv`:
- **id**: Unique identifier for each product.
- **gender**: Gender category (e.g., Male, Female)
- **masterCategory**: High-level product category (e.g., Apparel, Accessories)
- **subCategory**: More specific product category (e.g., Topwear, Bottomwear)
- **articleType**: Type of article (e.g., Shirt, Jeans)
- **baseColour**: Primary color of the product
- **season**: Season associated with the product (e.g., Summer, Winter)
- **usage**: Intended usage (e.g., Casual, Formal)
- **year**: Year of product release

## Data Visualization 📈

We explore the dataset and visualize various aspects using different types of plots. Below are the types of visualizations we have created and the corresponding images:

### 1. Line Plot
- **Description**: Shows the trend of fashion products over the years.
- **Image**: ![Line Plot]![image](https://github.com/AbdelrahmanHassan111/Intelligent-product-Recommendation-System/assets/156480367/effc9c1b-ebbc-4856-b378-466104dadbde)

### 2. Bar Plot
- **Description**: Displays the distribution of master categories.
- **Image**: ![Bar Plot]![image](https://github.com/AbdelrahmanHassan111/Intelligent-product-Recommendation-System/assets/156480367/7c2aef7d-3bc4-4dc7-aab9-a8fccaf91c8d)


### 3. Histogram
- **Description**: Illustrates the distribution of products by year.
- **Image**: ![Histogram]![image](https://github.com/AbdelrahmanHassan111/Intelligent-product-Recommendation-System/assets/156480367/3f86e035-23d4-4c4c-b2ad-1d9e8755a80b)


### 4. Pie Chart
- **Description**: Represents the gender distribution of products.
- **Image**: ![Pie Chart]![image](https://github.com/AbdelrahmanHassan111/Intelligent-product-Recommendation-System/assets/156480367/5466beca-00c1-41b6-aeaf-2e4200ef6b8e)


### 5. Scatter Plot
- **Description**: Plots the relationship between product ID and year.
- **Image**: ![Scatter Plot]![image](https://github.com/AbdelrahmanHassan111/Intelligent-product-Recommendation-System/assets/156480367/0b599e6f-a1d5-45c6-8982-10de49223e2f)


### 6. Box Plot
- **Description**: Shows the year distribution across different master categories.
- **Image**: ![Box Plot]![image](https://github.com/AbdelrahmanHassan111/Intelligent-product-Recommendation-System/assets/156480367/20bbfdd9-3b1f-4527-928a-6a90d22fb26e)


### 7. Pair Plot
- **Description**: Illustrates pairwise relationships between features.
- **Image**: ![Pair Plot]![image](https://github.com/AbdelrahmanHassan111/Intelligent-product-Recommendation-System/assets/156480367/c20afbdd-824e-497e-b82b-5b6f4dc9136c)

## Model Training and Evaluation 🤖

We implemented a deep learning model using TensorFlow to classify the product images into different categories. Below is an overview of the model architecture and training process.

### Model Architecture
- **Base Model**: We used ResNet50, a deep convolutional neural network pre-trained on the ImageNet dataset, as the base model. This model is known for its accuracy in image classification tasks.
- **Custom Layers**: On top of the ResNet50 base, we added a global average pooling layer, a dropout layer to prevent overfitting, and a dense layer with softmax activation for classification into multiple categories.

### Data Preprocessing
- **Image Preprocessing**: Images were resized to 60x80 pixels and normalized to have pixel values between 0 and 1.
- **Data Augmentation**: Techniques like rescaling were used to increase the diversity of the training data and help the model generalize better.

### Training Process
- **Optimizer**: We used the Adam optimizer, which is known for its efficiency in training deep neural networks.
- **Loss Function**: Categorical Crossentropy was used as the loss function, appropriate for multi-class classification problems.
- **Metrics**: Accuracy was tracked during training to evaluate the model's performance.
- **Epochs**: The model was trained for 15 epochs.

### Evaluation
- **Validation**: A separate validation dataset was used to monitor the model's performance during training and prevent overfitting.
- **Test Accuracy**: After training, the model was evaluated on a test dataset to assess its accuracy and generalization capability.

## Testing and Deployment 🧪🚀

For testing the model, we developed a Streamlit app that allows users to upload an image and get predictions along with recommendations.

### Testing Process
- **Model Loading**: The trained model is loaded using TensorFlow.
- **Image Upload**: Users can upload an image through the Streamlit interface.
- **Preprocessing**: The uploaded image is preprocessed to match the format expected by the model.
- **Prediction**: The model predicts the category of the uploaded image.
- **Recommendations**: Based on the predicted category, the app fetches and displays similar product images.

### Streamlit App Features
- **User Interface**: A simple and interactive UI for uploading images and viewing results.
- **Recommendations**: Displays top 5 recommended images from the predicted category.
- **Probability Display**: Shows the probability of the most similar image.

## Conclusion 📜

This project demonstrates the application of deep learning in classifying e-commerce products and providing recommendations based on product images. The combination of data visualization, model training, and interactive testing showcases the potential of machine learning in the e-commerce domain.

## Repository Structure 📂

- **data/**: Contains the `styles.csv` file and the `images/` folder with product images.
- **notebooks/**: Jupyter notebooks for data exploration, visualization, and model training.
- **app/**: Streamlit app code for testing the model.
- **images/**: Contains images of the plots generated during data visualization.
- **models/**: Saved model files.
- **README.md**: Project overview and documentation.

## Authors 👥

- [Your Name](https://github.com/yourusername)
- [Collaborator Name](https://github.com/collaboratorusername)

Feel free to contribute to this project by submitting issues or pull requests. Let's make e-commerce smarter with AI! 🚀
