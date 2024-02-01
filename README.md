# Breast-Cancer-Detection-using-IR-Image
Breast Cancer Detection using Infrared (IR) Images is crucial for early diagnosis. Traditional methods face limitations, prompting the utilization of advanced machine learning techniques.In this project, a hybrid model incorporating Convolutional Neural Networks (CNN) and Support Vector Machines (SVM) is employed. IR grayscale images, sourced from Kaggle's dataset with folders for healthy and unhealthy samples, offer a unique perspective. With 30% dataset utilization for end detection, the project aims to enhance accuracy and streamline the diagnostic process. The integration of CNN for feature extraction and SVM for classification underscores the potential impact on improving breast cancer diagnostic precision.
 
# Problem Statement
01:-Diagnostic Limitations: Traditional methods struggle with accurate early-stage breast cancer detection.

02:-Time Constraints: Diagnostic processes are time-consuming, impacting timely treatment decisions.

03:-Accuracy Challenges: Current technologies may yield false positives/negatives, affecting diagnostic precision.

04:-Limited Accessibility: Unequal access to diagnostic technologies poses challenges, especially in resource-limited settings.

# Data Collection & Preparation
You want the dataset which is used in this project. conntact me throught email. Email is mention in my profile"

Dataset Overview:
- The dataset comprises two folders: "healthy" and "unhealthy."   - The "healthy" folder contains approximately 642 grayscale images.   - The "unhealthy" folder contains approximately 640   grayscale images.

Image Characteristics:
- All images in the dataset are in grayscale format.   - The grayscale images provide a    monochromatic representation of the data.

Dataset Size:
- The entire dataset has a size of 540 megabytes, providing a substantial amount of data for analysis.

Origin of Data
- The dataset originates from a Portuguese hospital's data, contributing to its relevance and potential clinical insights.

# Model Related
Models Utilized:
- Three models were employed in the project: Convolutional Neural Network (CNN), Support Vector Machine (SVM), and a Confusion Matrix for evaluation.
  
CNN Model Creation:
- The CNN model was created as a crucial component of the project.During the process, overfitting of the output was identified and addressed.
  
Overfitting Mitigation:
- Due to overfitting in the CNN output, appropriate measures were taken to mitigate this issue.The CNN implementation leveraged scikit-learn's built-in functionalities to enhance robustness.
  
Feature Extraction and Model Saving:
- Features were extracted from the CNN model, capturing essential patterns.The CNN model, along with its learned features, was saved for future use.
  
SVM for Classification:
- The saved features were utilized to train a Support Vector Machine (SVM) for classification.
  
High Accuracy Achieved:
- The combined CNN-SVM model yielded impressive results, achieving an accuracy of 98%.
Confusion Matrix for Evaluation:A confusion matrix was employed to assess the model's performance and predictions.


# Conclusion
The project employed a three-pronged approach, utilizing Convolutional Neural Network (CNN), Support Vector Machine (SVM), and Confusion Matrix. Despite facing overfitting challenges in the CNN model output, strategic exclusions and scikit-learn library integration were leveraged. Feature extraction from the modified CNN yielded crucial patterns, contributing to a subsequent SVM classification achieving an impressive 98% accuracy. The comprehensive assessment, including Confusion Matrix analysis, demonstrated the project's holistic and successful integration of multiple models for effective breast cancer detection.

# Future Scope
Beyond the current project, there's immense potential for advancing breast cancer detection using infrared (IR) images. Data preprocessing plays a pivotal role, presenting avenues for further exploration. Implementing sophisticated preprocessing techniques, such as noise reduction and image enhancement, could enhance the model's sensitivity and specificity. Integration of additional datasets from diverse sources may broaden the model's applicability and robustness. Furthermore, exploring novel deep learning architectures and leveraging transfer learning methodologies could amplify the model's feature extraction capabilities. Continuous refinement in data preprocessing methodologies stands as a promising avenue for optimizing the accuracy and reliability of breast cancer detection models.



