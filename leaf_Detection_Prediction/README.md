#  Disease Diagnosis: A Multimodal CNN-LSTM for Farms

<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/305fd4a9-bea5-45b6-974c-14bde249c00c" 
    alt="image" 
    width="400" 
    height="350" 
    style="border-radius:25px;"
  />
</p>

A groundbreaking **multimodal deep learning system** that combines computer vision with environmental context to deliver highly accurate agricultural disease diagnosis. This innovative approach solves the critical "ambiguity problem" in plant pathology by integrating visual symptoms with weather patterns.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)



##  Project Objective

Build a highly accurate **"in-the-wild" disease classifier** that solves the fundamental **"ambiguity problem"** in agricultural diagnostics. For example, a yellow spot on a leaf could be:

- **Fungal Infection** (like Rust, which thrives in high humidity)  
- **Nutrient Deficiency** (unrelated to weather conditions)

By feeding the model both the **image** and **recent weather data**, it learns to make intelligent, context-aware diagnoses that traditional single-modal systems cannot achieve.



##  The "Untouched" Hybrid Architecture

This is not just another CNN or LSTM project. It's a **unified multimodal network** with two specialized branches:

### CNN Branch (Visual Analysis)
- **Input**: Real-field crop photographs
- **Model**: Pre-trained CNN (MobileNetV2/ResNet)
- **Output**: Feature vector describing visual patterns (spots, colors, textures)

### LSTM Branch (Contextual Analysis)  
- **Input**: 7-14 days of historical weather data
- **Model**: LSTM/RNN network
- **Output**: Context vector summarizing environmental trends
 <p align="center">
  <img 
    src="https://github.com/user-attachments/assets/3f7ee077-4041-4256-8a64-df43617b2867" 
    alt="image" 
    width="300" 
    height="300" 
    style="border-radius:25px;"
  />
</p> 

### Fusion & Classification
- **Fusion**: Concatenate visual + context vectors
- **Decision**: Final classification using dense layers
- **Output**: Context-aware diagnosis (Rice Blast, Sheath Blight, Nutrient Deficiency, Healthy)



### Technical Flow:
1. **Image Processing**: CNN extracts visual features → 1×1024 vector  
2. **Weather Processing**: LSTM analyzes temporal patterns → 1×128 vector  
3. **Multimodal Fusion**: Concatenate vectors → 1×1152 combined representation  
4. **Context-Aware Classification**: Final diagnosis based on complete situational understanding



## Data Sources & Collection

### Image Data
- **Public Datasets**: Kaggle, GitHub repositories (Rice, Wheat diseases)
- **Field Collection**: 100-200 "in-the-wild" photos from local farms/markets
- **Annotation**: Manual labeling with expert validation

### Environmental Data  
- **Primary Source**: ICAR-CRIDA Crop-Pest-Disease-Weather Database
- **Field Integration**: GPS location + timestamp for each field photo
- **API Supplement**: OpenWeatherMap historical data (14-day weather reconstruction)

### The Real Innovation: Data Pairing
The most challenging and novel aspect is creating a **curated multimodal dataset** where each field image is perfectly paired with its corresponding 14-day weather sequence for that specific location and date.



## Implementation Highlights

### Core Features
- **Multimodal Intelligence**: Visual + contextual data fusion  
- **Real-World Validation**: Field-tested with Indian farm conditions  
- **Temporal Awareness**: 14-day weather pattern analysis  
- **Ambiguity Resolution**: Differentiates visually similar conditions  
- **Scalable Architecture**: Adaptable to multiple crops and regions  

### Technical Stack
- **Deep Learning**: PyTorch/TensorFlow for model development  
- **Computer Vision**: OpenCV for image preprocessing  
- **Data Processing**: Pandas, NumPy for weather data handling  
- **Geospatial**: GPS integration for location-specific weather  
- **APIs**: RESTful services for historical weather data  



##  Expected Outcomes

### Diagnostic Accuracy
| Condition | Traditional CNN | Our Multimodal Approach |
|-----------|-----------------|-------------------------|
| Rust vs Deficiency | 75% accuracy | **92%+ accuracy** |
| False Positives | High | **Significantly Reduced** |
| Real-World Performance | Limited | **Field-Ready** |

### Agricultural Impact
- **Reduced pesticide misuse** through accurate diagnosis  
- **Early disease detection** using weather patterns  
- **Cost savings** for small-scale Indian farmers  
- **Sustainable farming** practices through precise interventions  



##  Contributing

We welcome contributions from researchers, developers, and agricultural experts! This project addresses a critical need in sustainable farming and there are many opportunities for enhancement.

### Areas for Contribution:
- **Data Collection**: Help expand our field image dataset with more crops and regions  
- **Model Improvements**: Optimize the CNN-LSTM architecture or explore new fusion techniques  
- **API Integration**: Enhance weather data pipelines and add soil data sources  
- **Mobile Deployment**: Help create farmer-friendly mobile applications  
- **Documentation**: Improve tutorials and multilingual support for Indian farmers  

### How to Contribute:
1. Fork the repository  
2. Create your feature branch: `git checkout -b feature/AmazingFeature`  
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`  
4. Push to the branch: `git push origin feature/AmazingFeature`  
5. Open a Pull Request  



##  Developer

**Naveen Kumar S**  
 Passionate about AI in Agriculture & Sustainable Farming Solutions  

-  GitHub: [Naveenr810953](https://github.com/Naveenr810953)



##  Show Your Support

If this project helps advance sustainable agriculture or inspires your work, please give it a **star** on GitHub! Your support helps bring AI-powered farming solutions to more Indian farmers.

**"Together, let's build intelligent systems that understand both crops and context!"**
